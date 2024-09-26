from audiocraft.metrics import CLAPTextConsistencyMetric
from huggingface_hub import hf_hub_download
import bittensor as bt
import numpy as np
import torchaudio
import librosa
import torch


class MetricEvaluator:
    @staticmethod
    def calculate_snr(file_path, silence_threshold=1e-4, constant_signal_threshold=1e-2):
        """Calculates the Signal-to-Noise Ratio (SNR) of the audio signal."""
        audio_signal, _ = librosa.load(file_path, sr=None)
        
        # Check for silence or constant signal
        if np.max(np.abs(audio_signal)) < silence_threshold or np.var(audio_signal) < constant_signal_threshold:
            return -np.inf
        
        signal_power = np.mean(audio_signal**2)
        noise_signal = librosa.effects.preemphasis(audio_signal)
        noise_power = np.mean(noise_signal**2)
        
        if noise_power < 1e-10:
            return np.inf
        
        snr = 10 * np.log10(signal_power / noise_power)
        return snr

    @staticmethod
    def calculate_consistency(file_path, text):
        """Calculates the consistency between the generated music and the text prompt using CLAPTextConsistencyMetric."""
        try:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            pt_file = hf_hub_download(repo_id="lukewys/laion_clap", filename="music_audioset_epoch_15_esc_90.14.pt")
            clap_metric = CLAPTextConsistencyMetric(pt_file, model_arch='HTSAT-base').to(device)

            def convert_audio(audio, from_rate, to_rate, to_channels):
                """Resamples the audio to the target sample rate and converts it to mono if necessary."""
                resampler = torchaudio.transforms.Resample(orig_freq=from_rate, new_freq=to_rate)
                audio = resampler(audio)
                if to_channels == 1:
                    audio = audio.mean(dim=0, keepdim=True)
                return audio

            # Load and resample the audio
            audio, sr = torchaudio.load(file_path)
            audio = convert_audio(audio, from_rate=sr, to_rate=sr, to_channels=1)

            # Update the CLAP metric with the audio and text
            clap_metric.update(audio.unsqueeze(0), [text], torch.tensor([audio.shape[1]]), torch.tensor([sr]))
            consistency_score = clap_metric.compute()
            return consistency_score
        
        except Exception as e:
            bt.logging.error(f"Error calculating music consistency score: {e}")
            return None


class MusicQualityEvaluator:
    def __init__(self):
        pass

    def evaluate_music_quality(self, file_path, text=None):
        """Evaluates the overall quality of the generated music using SNR and text consistency metrics."""
        # Initialize metrics
        try:
            snr_score = MetricEvaluator.calculate_snr(file_path)
            bt.logging.info(f'SNR: {snr_score} dB')
        except Exception as e:
            bt.logging.error(f"Failed to calculate SNR: {e}")
            snr_score = -np.inf  # Default to a very low value

        try:
            consistency_score = MetricEvaluator.calculate_consistency(file_path, text)
            bt.logging.info(f'Consistency Score: {consistency_score}')
        except Exception as e:
            bt.logging.error(f"Failed to calculate consistency score: {e}")
            consistency_score = None

        # Normalize SNR score
        normalized_snr = 1 / (1 + np.exp(-snr_score / 20)) if snr_score != -np.inf else 0

        # Normalize consistency score (scaling from [-1, 1] to [0, 1])
        normalized_consistency = (consistency_score + 1) / 2 if consistency_score is not None and consistency_score >= 0 else 0

        # Log normalized metrics
        bt.logging.info(f'Normalized Metrics: SNR = {normalized_snr}, Consistency = {normalized_consistency}')

        # Aggregate score: 60% weight for SNR, 40% weight for consistency
        aggregate_score = 0.6 * normalized_snr + 0.4 * normalized_consistency

        # Penalize the score if the consistency score is too low (< 0.2)
        if consistency_score is None or consistency_score < 0.2:
            aggregate_score = 0

        bt.logging.info(f'Aggregate Score: {aggregate_score}')
        return aggregate_score