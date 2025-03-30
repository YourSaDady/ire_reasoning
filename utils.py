import time
import torch
from transformers import AutoTokenizer
import json

def check_grad(model):
    '''
    check whether the model is generating gradient and doing back-prop
    '''
    # Access gradients for the model's parameters
    gradients_model = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            gradients_model[name] = param.grad
    print("\nGradients for model's parameters:")
    for i, (name, grad) in enumerate(gradients_model.items()):
        print(f"{name}: {grad}")
        if i == 10: 
            break
    print('\n.\n.\n.')

def convert_time(start_t):
    spent = time.time() - start_t
    hrs = int(spent // 3600)
    spent %= 3600
    mins = int(spent // 60)
    secs = int(spent % 60)
    return [hrs, mins, secs]

def printTokenizerRange(tokenizer):
    '''
    show the range of input indices of a given Tokenizer, and print all indices of special_tokens
    '''
    special_tokens = tokenizer.special_tokens_map
    special_indices = [tokenizer.convert_tokens_to_ids(token) for token in special_tokens.values()]
    vocab_size = tokenizer.vocab_size
    print(f'The tokenizer "{tokenizer.name_or_path}" has regular tokens: [0-{vocab_size-1}, and special indices: {special_indices}.')
    print(f'special_tokens_map: {special_tokens}')


class VisualizeEBMs:
    '''
    Visualize the sequence of EBMs TODO: 模仿下IRED的visual!
    '''
    def __init__(self, task_config, visual_config, sampling_config):
        self.num_ebms=task_config['out_len']
        self.num_vars=task_config['out_len']
        self.num_classes=task_config['num_classes']
        self.num_times=sampling_config.steps
        self.time_step=visual_config.time_step #time unit
        self.ebms_log = self._init_ebms()


    def _init_ebms(self):
        '''
        简化版的logging: initialize a dict of the annealed sequence of EBMs of all time steps:
            features: can be saved in jsonl or png formats

        ebms_log = {
            time_step: time_step,
            ebm_1: {
                t1: {
                    'landscape': <2D landscape> of Shape(k, num_classes),
                    'avg_energy': energy (scalar),
                    'avg_loss': loss (scalar)
                }
                t2: <2D landscape>,
                ...
                T: <2D landscape>
            },
            ebm_2: {
                ...
            },
            ...
            ebm_K: {...}
        }
        '''
        ebms_log = {}
        ebms_log['time_step'] = self.time_step
        for k in range(1, self.num_ebms+1):
            k_key = f'ebm_{k}'
            ebms_log[k_key] = {}
            for t in range(0, self.num_times, self.time_step):
                t_key = f't{t}' 
                ebms_log[k_key][t_key] = []
        return ebms_log

    def screenshot(self, k, t, cond_landscape, avg_loss):
        '''
        Insert the screenshot for the k-th energy landscape at time t

        params: 
            k: k-th EBM
            t: time t
            cond_landscape (Size(k, num_classes)): the 2D landscape of q(x_i | x_-i), where dim0 is x_i, dim1 is class_index
        '''
        k_key = f'ebm_{k}'
        t_key = f't{t}'
        avg_energy = torch.mean(cond_landscape).item()
        self.ebms_log[k_key][t_key] = {
            'landscape': cond_landscape.cpu().tolist(),
            'avg_energy': avg_energy,
            'avg_loss': avg_loss
        }

    # def show(self):
    #     #TODO: visualize
    #     return
        
    # def save(self, save_prefix):
    #     save_path= f'{save_prefix}/visualize.jsonl'
    #     with open(save_path)


def main():
    '''
    llama3.1-8b-instruct: 
        - regular tokens: [0-127999, and special indices: [128000, 128009]
        - special_tokens_map: {'bos_token': '<|begin_of_text|>', 'eos_token': '<|eot_id|>'}
    '''
    tok = AutoTokenizer.from_pretrained('meta-llama/Llama-3.1-8B-Instruct')
    printTokenizerRange(tok)

if __name__ == '__main__':
    main()