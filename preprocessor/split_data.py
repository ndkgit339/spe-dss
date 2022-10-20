import random
import hydra
from omegaconf import DictConfig

def split_data(utt_list_path, utt_list_path_train, utt_list_path_dev, 
               utt_list_path_synth, utt_list_path_eval, 
               utt_list_path_traindeveval, corpus_name, 
               split_rates):
    print("split data...")

    # Read utt list
    with open(utt_list_path, "r") as f:
        utt_list = [l.strip() for l in f]

    # Split rate
    r1 = split_rates[0]
    r2 = r1 + split_rates[1]
    r3 = r2 + split_rates[2]
    r4 = r3 + split_rates[3]

    if corpus_name == "JVS":
        speaker_ids = list(set([utt.split(":")[0] for utt in utt_list]))

        train_utts = []
        dev_utts = []
        synth_utts = []
        eval_utts = []
        traindeveval_utts = []
        for speaker_id in speaker_ids:
            utts = [utt for utt in utt_list if utt.startswith(speaker_id)]
            random.shuffle(utts)

            train_utts += utts[:r1]
            dev_utts += utts[r1:r2]
            synth_utts += utts[r2:r3]
            eval_utts += utts[r3:r4]
            traindeveval_utts += utts[:r4]
    
    else:
        raise ValueError("corpus_name only support \"JVS\"")


    with open(utt_list_path_train, "w") as f:
        f.write("\n".join(train_utts))

    with open(utt_list_path_dev, "w") as f:
        f.write("\n".join(dev_utts))

    with open(utt_list_path_synth, "w") as f:
        f.write("\n".join(synth_utts))

    with open(utt_list_path_eval, "w") as f:
        f.write("\n".join(eval_utts))

    with open(utt_list_path_traindeveval, "w") as f:
        f.write("\n".join(traindeveval_utts))

@hydra.main(config_path="conf/preprocess", config_name="config")
def main(config: DictConfig):

    split_data(config)
    
if __name__ == "__main__":
    main()
