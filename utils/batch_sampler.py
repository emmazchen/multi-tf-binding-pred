import random
import sys

"""
Works on sorted dataset sorted by len(x) in ascending order
"""
class CustomBatchSampler():
    def __init__(self, sampler, dataset, max_token_num, batch_size, drop_last=False): #take out batch_size later
        self.max_token_num = max_token_num
        self.sampler=sampler
        self.dataset=dataset
        self.indices = [_ for _ in range(len(dataset))] #for keeping track of unsampled indices
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self):
        while len(self.indices) != 0:
            i = random.randint(0, len(self.indices)-1) # index of indices
            idx = self.indices[i] #actual index of sequences
            token_num = len(self.dataset[idx][0])  # token_num is len(x) of first sampled seq
            n = int(self.max_token_num/pow(token_num,2)) # number of sequences to be in batch
            if n==0:
                print("Error: found sequence with length^2 exceeding max number of tokens allowed per batch", file=sys.stderr)

            batch=[]
            for _ in range(i-n+1, i+1):
                if _ < 0: # prevent wrap around
                    continue
                batch.append(self.indices[_])
            for _ in batch:
                self.indices.remove(_)
            yield batch
            batch=[]
            
        self.indices = [_ for _ in range(len(self.dataset))]


