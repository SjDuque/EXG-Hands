import numpy as np
import time
import pylsl
from itertools import combinations
import threading

class PromptServer(threading.Thread):
    def __init__(self, prompt_switch_interval=5):
        super().__init__()
        self.prompt_switch_interval = prompt_switch_interval
        self.prompt_labels = ['thumb', 'index', 'middle', 'ring', 'pinky']
        self.prompt_labels_idx = {label: i for i, label in enumerate(self.prompt_labels)}
        self.prompt_groups = ['thumb', 'index', 'middle', ['ring', 'pinky']]
        self.prompt_lists = self.generate_prompt_lists()
        self.prompt_index = -1
        self.running = False

        # LSL setup
        self.prompt_info = pylsl.StreamInfo(
            'finger_prompt', 'Markers', len(self.prompt_labels),
            1 / self.prompt_switch_interval, 'int8', 'finger_prompt')

        channels = self.prompt_info.desc().append_child("channels")
        for name in self.prompt_labels:
            channels.append_child("channel").append_child_value("label", name)

        self.prompt_outlet = pylsl.StreamOutlet(self.prompt_info)
        
        print("Labels:", self.prompt_labels)

    def generate_prompt_lists(self):
        result = []
        num_groups = len(self.prompt_groups)

        for num_true in range(num_groups + 1):
            for combo in combinations(range(num_groups), num_true):
                prompt = [False] * len(self.prompt_labels)
                for group_index in combo:
                    group = self.prompt_groups[group_index]
                    if isinstance(group, list):
                        for label in group:
                            prompt[self.prompt_labels_idx[label]] = True
                    else:
                        prompt[self.prompt_labels_idx[group]] = True
                result.append(prompt)
        return result

    def run(self):
        self.running = True
        current_prompt = self.prompt_lists[0] if self.prompt_lists else [False] * 5

        while self.running:
            prompt_int = [int(prompt) for prompt in current_prompt]
            self.prompt_outlet.push_sample(prompt_int)
            print(f"Sent prompt list: {prompt_int}")

            time.sleep(self.prompt_switch_interval)

            self.prompt_index = (self.prompt_index + 1) % len(self.prompt_lists)
            if self.prompt_index == 0:
                np.random.shuffle(self.prompt_lists)
            current_prompt = self.prompt_lists[self.prompt_index]


    def stop(self):
        self.running = False

if __name__ == "__main__":
    streamer = PromptServer()
    streamer.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Stopping PromptServer...")
        streamer.stop()
        streamer.join()
