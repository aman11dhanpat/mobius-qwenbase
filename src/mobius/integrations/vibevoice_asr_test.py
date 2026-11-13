# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for VibeVoice-ASR host processing and staged audio orchestration."""

from __future__ import annotations

import numpy as np
import pytest

from mobius.integrations.vibevoice_asr import (
    VibeVoiceASRBatch,
    VibeVoiceASRHost,
    VibeVoiceASRProcessor,
)


def test_processor_normalizes_pads_chunks_prompts_and_parses_diarization():
    processor = VibeVoiceASRProcessor()
    short = np.array([1.0, -1.0], dtype=np.float32)
    long = np.ones(processor.chunk_samples + 1, dtype=np.float32)
    batch = processor.prepare_audio([short, long], sampling_rate=24_000)

    assert batch.input_values.shape == (2, processor.chunk_samples + 1)
    assert batch.padding_mask[0].sum() == 2
    assert batch.padding_mask[1].sum() == processor.chunk_samples + 1
    # The normalized sine-like source has the source target RMS level.
    assert np.sqrt(np.mean(np.square(batch.input_values[0, : short.size]))) == pytest.approx(
        10 ** (-25 / 20)
    )
    chunks = list(processor.iter_chunks(batch))
    assert [chunk.input_values.shape for chunk in chunks] == [
        (2, 1, processor.chunk_samples),
        (2, 1, 1),
    ]
    messages = processor.make_prompt(audio_samples=6_401, context_info="Mobius is a hotword.")
    assert messages[1]["content"] == (
        "<|object_ref_start|><|box_start|><|box_start|><|box_start|><|object_ref_end|>\n"
        "This is a 0.27 seconds audio, with extra info: Mobius is a hotword.\n\n"
        "Please transcribe it with these keys: Start time, End time, Speaker ID, Content"
    )
    assert processor.make_prompt(audio_samples=3_200, context_info="  ")[1]["content"] == (
        "<|object_ref_start|><|box_start|><|object_ref_end|>\n"
        "This is a 0.13 seconds audio, please transcribe it with these keys: "
        "Start time, End time, Speaker ID, Content"
    )
    impulse = np.zeros(10_000, dtype=np.float32)
    impulse[0] = 1.0
    peak = np.abs(processor.prepare_audio(impulse, sampling_rate=24_000).input_values).max()
    rms = np.sqrt(np.mean(np.square(impulse), dtype=np.float64))
    unclipped_peak = np.float32(
        10 ** ((processor.target_dbfs - 20 * np.log10(rms + processor.eps)) / 20)
    )
    assert peak == pytest.approx(unclipped_peak / (unclipped_peak + processor.eps))
    assert processor.parse_diarization(
        'assistant\n[{"Start time": 0.0, "End_Time": 1.2, '
        '"Speaker ID": "S0", "Content": "hello"}]\n'
    ) == [{"start_time": 0.0, "end_time": 1.2, "speaker_id": "S0", "text": "hello"}]
    assert processor.parse_diarization(
        '[{"Start": 1.2, "End": 2.4, "Speaker": "S1", "Text": "world"}]'
    ) == [{"start_time": 1.2, "end_time": 2.4, "speaker_id": "S1", "text": "world"}]
    assert processor.parse_diarization(
        'assistant\n[{"Start": 0.0, "End": 1.0, "Speaker": "S0", '
        '"Content": "literal [ bracket"}]\n'
    ) == [
        {"start_time": 0.0, "end_time": 1.0, "speaker_id": "S0", "text": "literal [ bracket"}
    ]


def test_processor_rejects_wrong_rate_and_malformed_diarization():
    processor = VibeVoiceASRProcessor()
    with pytest.raises(ValueError, match="24000 Hz"):
        processor.prepare_audio(np.ones(10, dtype=np.float32), sampling_rate=16_000)
    with pytest.raises(ValueError, match="not valid diarization JSON"):
        processor.parse_diarization("not-json")
    with pytest.raises(ValueError, match="missing speaker_id"):
        processor.parse_diarization('[{"Start time": 0, "End time": 1, "Content": "hello"}]')


def test_host_preserves_chunk_cache_and_uses_seeded_connector_noise():
    processor = VibeVoiceASRProcessor()
    processor.chunk_samples = 4
    processor.hop_length = 4
    calls: list[tuple[str, dict[str, np.ndarray]]] = []

    def initial_cache(stage: str, batch_size: int) -> dict[str, np.ndarray]:
        assert stage in {"acoustic_encoder", "semantic_encoder"}
        return {"past_conv.0": np.zeros((batch_size, 1, 1), dtype=np.float32)}

    def run_stage(stage: str, feeds: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        calls.append((stage, feeds))
        if stage.endswith("encoder"):
            value = feeds["input_values"].mean(axis=-1, keepdims=False)[..., None]
            return {
                "audio_latents": value,
                "present_conv.0": feeds["past_conv.0"] + 1,
            }
        assert stage == "connectors"
        return {
            "audio_features": feeds["acoustic_latents"].reshape(-1, 1),
            "audio_feature_lengths": np.array([2], dtype=np.int64),
        }

    host = VibeVoiceASRHost(run_stage, initial_cache, processor)
    features, lengths = host.encode_audio(
        VibeVoiceASRBatch(
            input_values=np.ones((1, 6), dtype=np.float32),
            padding_mask=np.ones((1, 6), dtype=np.bool_),
        ),
        seed=9,
    )

    assert features.shape == (2, 1)
    assert lengths.tolist() == [2]
    # The second encoder window must consume the first window's state.
    assert calls[2][1]["past_conv.0"].item() == 1
    assert calls[2][1]["input_values"].shape[-1] == 2
    assert not calls[0][1]["is_final_chunk"]
    assert calls[2][1]["is_final_chunk"]
    connector_feeds = calls[-1][1]
    assert connector_feeds["acoustic_noise_scale"].shape == (1,)
    assert connector_feeds["acoustic_latent_noise"].shape == (1, 2, 1)


def test_host_finalizes_each_variable_length_utterance_at_its_own_endpoint():
    processor = VibeVoiceASRProcessor()
    processor.chunk_samples = 4
    processor.hop_length = 4
    calls: list[tuple[str, dict[str, np.ndarray]]] = []

    def initial_cache(_stage: str, batch_size: int) -> dict[str, np.ndarray]:
        return {"past_conv.0": np.zeros((batch_size, 1, 1), dtype=np.float32)}

    def run_stage(stage: str, feeds: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        calls.append((stage, feeds))
        if stage.endswith("encoder"):
            return {
                "audio_latents": feeds["input_values"].mean(axis=-1)[..., None],
                "present_conv.0": feeds["past_conv.0"] + 1,
            }
        return {
            "audio_features": feeds["acoustic_latents"].reshape(-1, 1),
            "audio_feature_lengths": np.array([feeds["acoustic_latents"].shape[1]]),
        }

    host = VibeVoiceASRHost(run_stage, initial_cache, processor)
    features, lengths = host.encode_audio(
        VibeVoiceASRBatch(
            input_values=np.array([[1.0, 1.0, 1.0, 1.0, 1.0, 0.0], [2.0] * 6]),
            padding_mask=np.array([[True] * 5 + [False], [True] * 6]),
        ),
        seed=9,
    )

    encoder_calls = [feeds for stage, feeds in calls if stage.endswith("encoder")]
    assert [feeds["input_values"].shape[-1] for feeds in encoder_calls] == [
        4,
        4,
        1,
        1,
        4,
        4,
        2,
        2,
    ]
    assert [bool(feeds["is_final_chunk"][0]) for feeds in encoder_calls] == [
        False,
        False,
        True,
        True,
        False,
        False,
        True,
        True,
    ]
    assert features.shape == (4, 1)
    assert lengths.tolist() == [2, 2]


def test_iter_chunks_yields_partial_final_window_when_length_does_not_divide():
    processor = VibeVoiceASRProcessor()
    chunk = 5
    processor.chunk_samples = chunk
    batch = VibeVoiceASRBatch(
        input_values=np.arange(12, dtype=np.float32).reshape(1, 12),
        padding_mask=np.ones((1, 12), dtype=np.bool_),
    )
    chunks = list(processor.iter_chunks(batch))
    assert [chunk_.input_values.shape for chunk_ in chunks] == [
        (1, 1, chunk),
        (1, 1, chunk),
        (1, 1, 2),
    ]
    assert [chunk_.padding_mask.shape for chunk_ in chunks] == [
        (1, chunk),
        (1, chunk),
        (1, 2),
    ]
    reassembled = np.concatenate([chunk_.input_values[:, 0, :] for chunk_ in chunks], axis=-1)
    assert (reassembled == batch.input_values[0]).all()
    assert chunks[-1].padding_mask.all()


def test_iter_chunks_keeps_windows_aligned_across_variable_length_batch():
    processor = VibeVoiceASRProcessor()
    processor.chunk_samples = 4
    batch = processor.prepare_audio(
        [np.arange(1.0, 6.0, dtype=np.float32), np.arange(1.0, 8.0, dtype=np.float32)],
        sampling_rate=24_000,
    )
    chunks = list(processor.iter_chunks(batch))
    assert [chunk_.input_values.shape for chunk_ in chunks] == [
        (2, 1, 4),
        (2, 1, 3),
    ]
    assert [chunk_.padding_mask.shape for chunk_ in chunks] == [(2, 4), (2, 3)]
    # Window 0: item 0 has 5 valid samples (4 + 1 padding tail), item 1 all valid.
    assert chunks[0].padding_mask[0].tolist() == [True, True, True, True]
    assert chunks[0].padding_mask[1].tolist() == [True, True, True, True]
    # Window 1: item 0 has 1 valid sample then padding; item 1 all valid.
    assert chunks[1].padding_mask[0].tolist() == [True, False, False]
    assert chunks[1].padding_mask[1].tolist() == [True, True, True]
    # Per-item samples are preserved end-to-end, padding intact.
    for index, valid in [(0, 5), (1, 7)]:
        reassembled = np.concatenate(
            [chunk_.input_values[index, 0, :] for chunk_ in chunks], axis=-1
        )
        assert (reassembled == batch.input_values[index]).all()
        assert batch.padding_mask[index, :valid].all()
        assert not batch.padding_mask[index, valid:].any()


def test_iter_chunks_rejects_batch_without_audio():
    processor = VibeVoiceASRProcessor()
    batch = VibeVoiceASRBatch(
        input_values=np.zeros((1, 0), dtype=np.float32),
        padding_mask=np.zeros((1, 0), dtype=np.bool_),
    )
    with pytest.raises(ValueError, match="no audio"):
        list(processor.iter_chunks(batch))


def test_processor_builds_source_chat_template_token_layout():
    class Tokenizer:
        def __init__(self):
            self.encoded_texts: list[str] = []

        @staticmethod
        def apply_chat_template(*args, **kwargs):
            raise AssertionError("ASR must not use the base tokenizer's native chat template")

        def encode(self, text):
            self.encoded_texts.append(text)
            if "<|im_start|>system\n" in text:
                return [1, 2]
            return [10, 11, *([42] * text.count("<|box_start|>")), 12]

        @staticmethod
        def convert_tokens_to_ids(token):
            assert token == "<|box_start|>"
            return 42

    tokenizer = Tokenizer()
    input_ids, acoustic_input_mask = VibeVoiceASRProcessor.build_input_ids(
        tokenizer,
        audio_samples=3_201,
        context_info="Alice",
    )

    assert input_ids == [1, 2, 10, 11, 42, 42, 12]
    assert acoustic_input_mask == [False, False, False, False, True, True, False]
    assert tokenizer.encoded_texts[0] == (
        "<|im_start|>system\n"
        "You are a helpful assistant that transcribes audio input into text output in JSON format."
        "<|im_end|>\n"
    )
    assert tokenizer.encoded_texts[1] == (
        "<|im_start|>user\n"
        "<|object_ref_start|><|box_start|><|box_start|><|object_ref_end|>\n"
        "This is a 0.13 seconds audio, with extra info: Alice\n\n"
        "Please transcribe it with these keys: Start time, End time, Speaker ID, Content"
        "<|im_end|>\n"
    )
