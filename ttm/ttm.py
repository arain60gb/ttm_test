from ttm.ttm_score import MusicQualityEvaluator
from ttm.protocol import MusicGeneration
from ttm.aimodel import AIModelService
from datasets import load_dataset
import bittensor as bt
import numpy as np
import torchaudio
import contextlib
import traceback
import asyncio
import random
import torch
import wandb
import wave
import sys
import os


# Set the project root path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
audio_subnet_path = os.path.abspath(project_root)
sys.path.insert(0, project_root)
sys.path.insert(0, audio_subnet_path)

class MusicGenerationService(AIModelService):
    def __init__(self):
        super().__init__()  
        self.load_prompts()
        self.total_dendrites_per_query = 10
        self.minimum_dendrites_per_query = 3  # Minimum dendrites per query
        self.current_block = self.subtensor.block
        self.last_updated_block = self.current_block - (self.current_block % 100)
        self.last_reset_weights_block = self.current_block
        self.filtered_axon = []
        self.combinations = []
        self.duration = 755  # 755 tokens = 15 seconds music
        self.lock = asyncio.Lock()

    def load_prompts(self):
        gs_dev = load_dataset("etechgrid/prompts_for_TTM")
        self.prompts = gs_dev['train']['text']
        return self.prompts

    async def run_async(self):
        step = 0
        while True:
            try:
                await self.main_loop_logic(step)
                step += 1
            except KeyboardInterrupt:
                print("Keyboard interrupt detected. Exiting MusicGenerationService.")
                break
            except Exception as e:
                print(f"An error occurred in MusicGenerationService: {e}")
                traceback.print_exc()

    async def main_loop_logic(self, step):
        g_prompt = None
        try:
            # Load prompt from the dataset using the load_prompts function
            bt.logging.info(f"Using prompt from HuggingFace Dataset for Text-To-Music at Step: {step}")
            g_prompt = self.load_prompts()
            g_prompt = random.choice(g_prompt)  # Choose a random prompt
            g_prompt = self.convert_numeric_values(g_prompt)
            print(f"____________________________TTM-Prompt selected____________________________:")
            # Ensure prompt length does not exceed 256 characters
            print(f"____________________________TTM-Prompt length____________________________: {len(g_prompt)}")
            print(f"____________________________TTM-Prompt length____________________________: {type(len(g_prompt))}")
            print(f"____________________________ TTM-Prompt ____________________________: {g_prompt}")

            try:
                while len(g_prompt) > 256:
                    print(f"____________________________ inside the promt ____________________________:")
                    bt.logging.error(f'The length of current Prompt is greater than 256. Skipping current prompt.')
                    g_prompt = random.choice(g_prompt)
                    print(f"____________________________TTM-Prompt randomly selected____________________________:")
                    g_prompt = self.convert_numeric_values(g_prompt)
            except Exception as e:
                bt.logging.error(f"An error occurred while checking prompt length check: {e}")

            # Get filtered axons and query the network
            filtered_axons = self.get_filtered_axons_from_combinations()
            bt.logging.info(f"______________TTM-Prompt______________: {g_prompt}")
            responses = self.query_network(filtered_axons, g_prompt)
            self.process_responses(filtered_axons, responses, g_prompt)

        except Exception as e:
            bt.logging.error(f"An error occurred in main loop logic: {e}")

    def query_network(self, filtered_axons, prompt):
        """Queries the network with filtered axons and prompt."""
        responses = self.dendrite.query(
            filtered_axons,
            MusicGeneration(text_input=prompt, duration=self.duration),
            deserialize=True,
            timeout=140,
        )
        return responses

    def process_responses(self, filtered_axons, responses, prompt):
        """Processes responses received from the network."""
        for axon, response in zip(filtered_axons, responses):
            if response is not None and isinstance(response, MusicGeneration):
                self.process_response(axon, response, prompt)
        
        bt.logging.info(f"Scores after update in TTM: {self.scores}")

    def process_response(self, axon, response, prompt):
        """Processes a single response from the network."""
        try:
            if response is not None and isinstance(response, MusicGeneration) and response.music_output:
                bt.logging.success(f"Received music output from {axon.hotkey}")
                self.handle_music_output(axon, response.music_output, prompt, response.model_name)
            else:
                self.punish(axon, service="Text-To-Music", punish_message="Invalid response")
        except Exception as e:
            bt.logging.error(f"Error processing response: {e}")

    def handle_music_output(self, axon, music_output, prompt, model_name):
        """Handles the received music output and saves it to file."""
        try:
            # Convert the list to a tensor
            audio_data = torch.Tensor(music_output) / torch.max(torch.abs(torch.Tensor(music_output)))

            # Convert to 32-bit PCM and save the file as .wav
            audio_data_int = (audio_data * 2147483647).type(torch.IntTensor).unsqueeze(0)
            output_path = os.path.join('/tmp', f'output_music_{axon.hotkey}.wav')
            torchaudio.save(output_path, src=audio_data_int, sample_rate=32000)
            bt.logging.info(f"Saved audio file to {output_path}")

            # Log audio to WandB
            uid_in_metagraph = self.metagraph.hotkeys.index(axon.hotkey)
            wandb.log({f"TTM prompt: {prompt[:100]} ....": wandb.Audio(np.array(audio_data), caption=f'For HotKey: {axon.hotkey[:10]}', sample_rate=32000)})
            bt.logging.success(f"TTM Audio file uploaded to wandb for Hotkey {axon.hotkey}")

            # Calculate the duration and adjust the score
            duration = self.get_duration(output_path)
            score = self.score_output(output_path, prompt)
            if duration < 15:
                score = self.score_adjustment(score, duration)

            bt.logging.info(f"Final score after processing: {score}")
            self.update_score(axon, score, service="Text-To-Music")

        except Exception as e:
            bt.logging.error(f"Error handling music output: {e}")

    def get_duration(self, wav_file_path):
        """Returns the duration of the audio file in seconds."""
        with contextlib.closing(wave.open(wav_file_path, 'r')) as f:
            frames = f.getnframes()
            rate = f.getframerate()
            return frames / float(rate)

    def score_adjustment(self, score, duration):
        """Adjusts the score based on the duration of the generated audio."""
        conditions = [
            (lambda d: 14.5 <= d < 15, 0.9),
            (lambda d: 14 <= d < 14.5, 0.8),
            (lambda d: 13.5 <= d < 14, 0.7),
            (lambda d: 13 <= d < 13.5, 0.6),
            (lambda d: 12.5 <= d < 13, 0.0),
        ]
        for condition, multiplier in conditions:
            if condition(duration):
                return score * multiplier
        return score

    def score_output(self, output_path, prompt):
        """Evaluates and returns the score for the generated music output."""
        try:
            score_object = MusicQualityEvaluator()
            return score_object.evaluate_music_quality(output_path, prompt)
        except Exception as e:
            bt.logging.error(f"Error scoring output: {e}")
            return 0.0

    def get_filtered_axons_from_combinations(self):
        """Gets filtered axons from the combinations."""
        if not self.combinations:
            self.get_filtered_axons()

        current_combination = self.combinations.pop(0)
        filtered_axons = [self.metagraph.axons[i] for i in current_combination]
        return filtered_axons

    def get_filtered_axons(self):
        # Get the uids of all miners in the network.
        uids = self.metagraph.uids.tolist()
        queryable_uids = (self.metagraph.total_stake >= 0)
        # Remove the weights of miners that are not queryable.
        queryable_uids = queryable_uids * torch.Tensor([self.metagraph.neurons[uid].axon_info.ip != '0.0.0.0' for uid in uids])
        # queryable_uid = queryable_uids * torch.Tensor([
        #     any(self.metagraph.neurons[uid].axon_info.ip == ip for ip in lib.BLACKLISTED_IPS) or
        #     any(self.metagraph.neurons[uid].axon_info.ip.startswith(prefix) for prefix in lib.BLACKLISTED_IPS_SEG)
        #     for uid in uids
        # ])
        active_miners = torch.sum(queryable_uids)
        dendrites_per_query = self.total_dendrites_per_query

        # if there are no active miners, set active_miners to 1
        if active_miners == 0:
            active_miners = 1
        # if there are less than dendrites_per_query * 3 active miners, set dendrites_per_query to active_miners / 3
        if active_miners < self.total_dendrites_per_query * 3:
            dendrites_per_query = int(active_miners / 3)
        else:
            dendrites_per_query = self.total_dendrites_per_query
        
        # less than 3 set to 3
        if dendrites_per_query < self.minimum_dendrites_per_query:
                dendrites_per_query = self.minimum_dendrites_per_query
        # zip uids and queryable_uids, filter only the uids that are queryable, unzip, and get the uids
        zipped_uids = list(zip(uids, queryable_uids))
        # zipped_uid = list(zip(uids, queryable_uid))
        filtered_zipped_uids = list(filter(lambda x: x[1], zipped_uids))
        filtered_uids = [item[0] for item in filtered_zipped_uids] if filtered_zipped_uids else []
        # filtered_zipped_uid = list(filter(lambda x: x[1], zipped_uid))
        # filtered_uid = [item[0] for item in filtered_zipped_uid] if filtered_zipped_uid else []
        # self.filtered_axon = filtered_uid

        subset_length = min(dendrites_per_query, len(filtered_uids))
        # Shuffle the order of members
        random.shuffle(filtered_uids)
        # Generate subsets of length 7 until all items are covered
        while filtered_uids:
            subset = filtered_uids[:subset_length]
            self.combinations.append(subset)
            filtered_uids = filtered_uids[subset_length:]
        return self.combinations


