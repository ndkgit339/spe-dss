from pathlib import Path
import hydra
from omegaconf import OmegaConf, DictConfig
from tqdm import tqdm

from matplotlib import pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

from parallel_wavegan.losses import stft, STFTLoss, MultiResolutionSTFTLoss, \
                                    DiscriminatorAdversarialLoss, \
                                    GeneratorAdversarialLoss
from dss import DifferentiableSpeechSynthesizer

# My library
from util_audio import show_spec, show_f0
from util_train import set_seed, decode_feats, world_analyze_synth, \
                       synth_from_coded_feats, calc_lsd, get_scaler_vals
from dataset import MyDataset


def get_data_loaders(data_config, data_dir, hop_length, n_frames,
                     n_frame_list_path):

    utt_n_frame_dict = {}
    with open(n_frame_list_path, "r") as f:
        for l in f:
            utt_n_frame_dict[l.strip().split(":")[0]] \
                = int(l.strip().split(":")[1])

    data_loaders = {}
    for phase in ["train", "dev", "synth"]:
        utt_list_path = data_dir / Path(data_config[phase].utt_list)
        with open(utt_list_path) as f:
            utt_list = [utt.strip() for utt in f if len(utt.strip()) > 0]

        # paths
        spectrogram_paths = [
            data_dir / Path(data_config["spectrogram"]) / \
                "{}-feats.npy".format(utt.replace(":", "-"))
            for utt in utt_list]
        f0_paths = [
            data_dir / Path(data_config["f0"]) / \
                "{}-feats.npy".format(utt.replace(":", "-"))
            for utt in utt_list]
        spectrum_paths = [
            data_dir / Path(data_config["spectrum"]) / \
                "{}-feats.npy".format(utt.replace(":", "-"))
            for utt in utt_list]
        aperiodicity_paths = [
            data_dir / Path(data_config["aperiodicity"]) / \
                "{}-feats.npy".format(utt.replace(":", "-"))
            for utt in utt_list]
        wav_paths = [
            data_dir / Path(data_config["wav"]) / \
                "{}-feats.npy".format(utt.replace(":", "-"))
            for utt in utt_list]

        # start frames
        start_frames = []
        for i, utt in enumerate(utt_list):
            if phase == "train" or phase =="dev":
                n_frame = utt_n_frame_dict[utt.replace(":", "-")]
                if n_frame < n_frames:
                    start_frames.append((i, 0))
                for start_frame in range(0, n_frame - n_frames + 1, 100):
                    start_frames.append((i, start_frame))

        batch_size = data_config[phase].batch_size
        dataset = MyDataset(phase, spectrogram_paths, f0_paths,
                            spectrum_paths, aperiodicity_paths,
                            wav_paths, start_frames, n_frames, hop_length)
        data_loaders[phase] = DataLoader(dataset, batch_size=batch_size,
                                         collate_fn=dataset.collate_fn,
                                         num_workers=data_config.num_workers,
                                         shuffle=phase.startswith("train"))

    return data_loaders


def get_model(optimizer_G_name, optimizer_G_params, 
              lr_scheduler_G_name, lr_scheduler_G_params, 
              config_model_D, optimizer_D_name, optimizer_D_params,
              lr_scheduler_D_name, lr_scheduler_D_params,
              restore_step, device, ckpt_dir, adv_train=False,
              training_mode="each_feats_model",
              config_model_G=None,
              config_model_G_f0=None, config_model_G_sp=None,
              config_model_G_ap=None, data_parallel=False):

    optimizer_class_G = getattr(torch.optim, optimizer_G_name)
    lr_scheduler_class_G = getattr(torch.optim.lr_scheduler, lr_scheduler_G_name)

    G, G_f0, G_sp, G_ap, optimizer_G, optimizer_G_f0, optimizer_G_sp, \
    optimizer_G_ap, lr_scheduler_G, lr_scheduler_G_f0, lr_scheduler_G_sp, \
    lr_scheduler_G_ap \
        = None, None, None, None, None, None, None, None, None, None, None, None

    if not training_mode in ["shared_model", "each_feats_model"]:
        ValueError("training mode has invalid value")

    if training_mode == "shared_model":
        G = hydra.utils.instantiate(config_model_G)
        if data_parallel:
            G = nn.DataParallel(G)
        G = G.to(device)
        optimizer_G = optimizer_class_G(G.parameters(), **optimizer_G_params)
        lr_scheduler_G = lr_scheduler_class_G(optimizer_G, **lr_scheduler_G_params)

    if training_mode == "each_feats_model":
        G_f0 = hydra.utils.instantiate(config_model_G_f0)
        G_sp = hydra.utils.instantiate(config_model_G_sp)
        G_ap = hydra.utils.instantiate(config_model_G_ap)
        if data_parallel:
            G_f0 = nn.DataParallel(G_f0)
            G_sp = nn.DataParallel(G_sp)
            G_ap = nn.DataParallel(G_ap)
        G_f0 = G_f0.to(device)
        G_sp = G_sp.to(device)
        G_ap = G_ap.to(device)
        optimizer_G_f0 = optimizer_class_G(G_f0.parameters(), **optimizer_G_params)
        optimizer_G_sp = optimizer_class_G(G_sp.parameters(), **optimizer_G_params)
        optimizer_G_ap = optimizer_class_G(G_ap.parameters(), **optimizer_G_params)
        lr_scheduler_G_f0 = lr_scheduler_class_G(optimizer_G_f0, **lr_scheduler_G_params)
        lr_scheduler_G_sp = lr_scheduler_class_G(optimizer_G_sp, **lr_scheduler_G_params)
        lr_scheduler_G_ap = lr_scheduler_class_G(optimizer_G_ap, **lr_scheduler_G_params)


    D, optimizer_D, lr_scheduler_D = None, None, None

    if adv_train:
        D = hydra.utils.instantiate(config_model_D)
        if data_parallel:
            D = nn.DataParallel(D)
        D = D.to(device)
        optimizer_class_D = getattr(torch.optim, optimizer_D_name)
        lr_scheduler_class_D = getattr(
            torch.optim.lr_scheduler, lr_scheduler_D_name)
        optimizer_D = optimizer_class_D(D.parameters(), **optimizer_D_params)
        lr_scheduler_D = lr_scheduler_class_D(
            optimizer_D, **lr_scheduler_D_params)

    if restore_step:
        ckpt_path = ckpt_dir / "step{}.pth.tar".format(restore_step)
        ckpt = torch.load(ckpt_path)
        if training_mode == "shared_model":
            G.load_state_dict(ckpt["G"])
            optimizer_G.load_state_dict(ckpt["optimizer_G"])
            lr_scheduler_G.load_state_dict(ckpt["lr_scheduler_G"])
        if training_mode == "each_feats_model":
            G_f0.load_state_dict(ckpt["G_f0"])
            G_sp.load_state_dict(ckpt["G_sp"])
            G_ap.load_state_dict(ckpt["G_ap"])
            optimizer_G_f0.load_state_dict(ckpt["optimizer_G_f0"])
            optimizer_G_sp.load_state_dict(ckpt["optimizer_G_sp"])
            optimizer_G_ap.load_state_dict(ckpt["optimizer_G_ap"])
            lr_scheduler_G_f0.load_state_dict(ckpt["lr_scheduler_G_f0"])
            lr_scheduler_G_sp.load_state_dict(ckpt["lr_scheduler_G_sp"])
            lr_scheduler_G_ap.load_state_dict(ckpt["lr_scheduler_G_ap"])
                
        if adv_train:
            D.load_state_dict(ckpt["D"])
            optimizer_D.load_state_dict(ckpt["optimizer_D"])
            lr_scheduler_D.load_state_dict(ckpt["lr_scheduler_D"])

    return G, G_f0, G_sp, G_ap, \
           optimizer_G, optimizer_G_f0, optimizer_G_sp, optimizer_G_ap, \
           lr_scheduler_G, lr_scheduler_G_f0, lr_scheduler_G_sp, lr_scheduler_G_ap, \
           D, optimizer_D, lr_scheduler_D


def get_speech_synthesizer_and_loss(device, sample_rate, n_fft, hop_length, synth_hop_length,
                             loss_name, loss_fft_size, loss_shift_size, 
                             loss_win_length, loss_type_G, loss_type_D,
                             data_parallel=False):

    # Loss
    speech_synthesizer = DifferentiableSpeechSynthesizer(
        device, sample_rate, n_fft, hop_length, synth_hop_length=synth_hop_length)
    if data_parallel:
        speech_synthesizer = nn.DataParallel(speech_synthesizer)
    speech_synthesizer = speech_synthesizer.to(device)

    criterion_stft = None
    if loss_name == "stft":
        criterion_stft = STFTLoss(
            fft_size=loss_fft_size,
            shift_size=loss_shift_size,
            win_length=loss_win_length)
    elif loss_name == "mrstft":
        criterion_stft = MultiResolutionSTFTLoss(
            fft_sizes=[1024, 2048, 512],
            hop_sizes=[120, 240, 50],
            win_lengths=[600, 1200, 240])

    criterion_gen_adv = GeneratorAdversarialLoss(
        loss_type=loss_type_G)
    criterion_dis_adv = DiscriminatorAdversarialLoss(
        loss_type=loss_type_D)

    if data_parallel:
        if loss_name != "only_feats":
            criterion_stft = nn.DataParallel(criterion_stft)
        criterion_gen_adv = nn.DataParallel(criterion_gen_adv)
        criterion_dis_adv = nn.DataParallel(criterion_dis_adv)
    if loss_name != "only_feats":
        criterion_stft = criterion_stft.to(device)
    criterion_gen_adv = criterion_gen_adv.to(device)
    criterion_dis_adv = criterion_dis_adv.to(device)

    return speech_synthesizer, criterion_stft, criterion_gen_adv, criterion_dis_adv


def get_loggers(log_dir):
    train_log_dir = log_dir / "train"
    val_log_dir = log_dir / "dev"
    synth_log_dir = log_dir / "synth"
    train_log_dir.mkdir(parents=True, exist_ok=True)
    val_log_dir.mkdir(parents=True, exist_ok=True)
    synth_log_dir.mkdir(parents=True, exist_ok=True)
    return (
        SummaryWriter(train_log_dir),
        SummaryWriter(val_log_dir),
        SummaryWriter(synth_log_dir)
    )


def logging(logger, current_step, device, speech_synthesizer, scaler_means, scaler_vars,
            basenames, in_spectrograms, 
            f0_hat,spectrum_hat,aperiodicity_hat,wav_hat,
            f0_targets, spectrum_targets,
            aperiodicity_targets, wav_targets, sample_rate, n_fft, hop_length,
            frame_shift_ms, comp, coding_ap, n_samples=1, mode="train"):

    n_samples = min(len(basenames), n_samples)
    width = 8 if mode == "train" else 20

    with torch.no_grad():
        for i in range(n_samples):
            basename = basenames[i]

            f0_target, spectrum_target, aperiodicity_target, pytorch_waveform_syn_coded = \
                synth_from_coded_feats(speech_synthesizer,
                    f0_targets[i], spectrum_targets[i], aperiodicity_targets[i],
                    scaler_means, scaler_vars, n_fft, sample_rate, device,
                    comp=comp, coding_ap=coding_ap)
            world_waveform_syn = world_analyze_synth(
                device, wav_targets[i], sample_rate, n_fft, hop_length)

            # f0
            fig_f0 = show_f0([f0_hat[i], f0_target.squeeze()],
                             ["output", "target"], 
                             frame_shift_ms, width=width)
            logger.add_figure(
                "{}/{}/F0".format(mode, basename), 
                fig_f0, global_step=current_step)
            plt.close()

            # スペクトル包絡
            fig_sp = show_spec([spectrum_hat[i], spectrum_target.squeeze()],
                               ["output", "target"],
                               frame_shift_ms=frame_shift_ms,
                               min_hz=0.0, max_hz=sample_rate / 2.0,
                               log=True, width=width)
            logger.add_figure(
                "{}/{}/log spectrum".format(mode, basename), 
                fig_sp, global_step=current_step)
            plt.close()

            ## 非周期性指標
            fig_ap = show_spec([aperiodicity_hat[i], aperiodicity_target.squeeze()],
                               ["output", "target"],
                               frame_shift_ms=frame_shift_ms, 
                               min_hz=0.0, max_hz=sample_rate / 2.0,
                               log=False, width=width)
            logger.add_figure(
                "{}/{}/aperiodicity".format(mode, basename), 
                fig_ap, global_step=current_step)
            plt.close()

            # スペクトログラム
            # in_spectrogram = torch.exp(
            #     in_spectrograms[i].to(device)
            #     * torch.sqrt(scaler_vars["spectrogram"].to(device))
            #     + scaler_means["spectrogram"].to(device)
            #     ).squeeze()
            gt_spec = stft(
                wav_targets[i].unsqueeze(0).to(device), 
                n_fft, hop_length, n_fft, 
                torch.hann_window(n_fft).to(device)
            ).transpose(2, 1).squeeze()

            out_mag_spec = stft(
                wav_hat[i].unsqueeze(0).to(device), 
                n_fft, hop_length, n_fft, 
                torch.hann_window(n_fft).to(device)
            ).transpose(2, 1).squeeze()
            pytorch_coded_mag_spec = stft(
                pytorch_waveform_syn_coded, 
                n_fft, hop_length, n_fft, 
                torch.hann_window(n_fft).to(device)
            ).transpose(2, 1).squeeze()
            world_mag_spec = stft(
                world_waveform_syn, 
                n_fft, hop_length, n_fft, 
                torch.hann_window(n_fft).to(device)
            ).transpose(2, 1).squeeze()

            fig_spec = show_spec(
                [out_mag_spec, pytorch_coded_mag_spec, world_mag_spec, gt_spec],
                ["output", "pytorch_coded", "world", "target"],
                frame_shift_ms=frame_shift_ms, 
                min_hz=0.0, max_hz=sample_rate / 2.0,
                log=True, width=width)
            logger.add_figure(
                "{}/{}/log spectrogram".format(mode, basename), 
                fig_spec, global_step=current_step)
            plt.close()

            # 音声
            logger.add_audio(
                "{}/{}/target_audio".format(mode, basename), 
                wav_targets[i].cpu().numpy(), 
                global_step=current_step, 
                sample_rate=sample_rate)
            logger.add_audio(
                "{}/{}/out_audio".format(mode, basename), 
                wav_hat[i].cpu().numpy(), 
                global_step=current_step, 
                sample_rate=sample_rate)
            logger.add_audio(
                "{}/{}/pytorch_coded_audio".format(mode, basename), 
                pytorch_waveform_syn_coded.squeeze(0).cpu().numpy(), 
                global_step=current_step, 
                sample_rate=sample_rate)
            logger.add_audio(
                "{}/{}/world_audio".format(mode, basename), 
                world_waveform_syn.squeeze(0).cpu().numpy(), 
                global_step=current_step, 
                sample_rate=sample_rate)


@hydra.main(config_path="conf", config_name="config")
def train(config: DictConfig):

    # Seed
    set_seed(config.seed)
    
    # Set in directory
    data_dir = Path(config.preprocess.data_dir)
    n_frame_list_path = Path(config.preprocess.data_dir) / "traindeveval_n_frame.list"

    # Set out directory
    out_dir = Path(config.train.out_dir)
    ckpt_dir = out_dir / "ckpt"
    log_dir = out_dir / "log"
    config_dir = out_dir / "config"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    config_dir.mkdir(parents=True, exist_ok=True)

    # Adversarial training
    adv_train = config.train.adversarial
    discriminate_step = config.train.discriminate_step
    discriminate_spec = config.train.discriminate_spec

    # Comp
    comp = config.preprocess.comp
    coding_ap = config.preprocess.audio.coding_ap

    # Save config
    with open(config_dir / "config.yaml", "w") as f:
        OmegaConf.save(config, f)

    # EPS
    eps = 1e-10

    # Get device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Audio params
    sample_rate = config.preprocess.audio.sample_rate
    n_fft = config.preprocess.audio.n_fft
    hop_length = config.preprocess.audio.hop_length
    synth_hop_length = config.preprocess.audio.synth_hop_length
    frame_shift_ms = 1000.0 * hop_length / sample_rate
    n_coded_sp = config.preprocess.audio.n_coded_sp
    n_coded_ap = config.preprocess.audio.n_coded_ap
    n_frames = config.preprocess.n_frames
    dim_sp = n_coded_sp if comp else n_fft // 2 + 1
    dim_ap = n_coded_ap if comp else n_fft // 2 + 1

    # Scaler val
    scaler_means, scaler_vars = get_scaler_vals(data_dir)

    # Data Loaders
    data_loaders = get_data_loaders(config.data, data_dir, hop_length, n_frames, n_frame_list_path)

    # Data parallel
    data_parallel = False
    if config.train.data_parallel and torch.cuda.device_count() > 1:
        print("--- Parallel training ---")
        data_parallel = config.train.data_parallel

    # Model, optimizer and lr scheduler
    restore_step = config.train.restore_step
    training_mode = config.train.training_mode
    G, G_f0, G_sp, G_ap, \
    optimizer_G, optimizer_G_f0, optimizer_G_sp, optimizer_G_ap, \
    lr_scheduler_G, lr_scheduler_G_f0, lr_scheduler_G_sp, lr_scheduler_G_ap, \
    D, optimizer_D, lr_scheduler_D = \
        get_model(config.train.optim.optimizer.G.name,
                  config.train.optim.optimizer.G.params,
                  config.train.optim.lr_scheduler.G.name,
                  config.train.optim.lr_scheduler.G.params,
                  config.model_D,
                  config.train.optim.optimizer.D.name,
                  config.train.optim.optimizer.D.params,
                  config.train.optim.lr_scheduler.D.name,
                  config.train.optim.lr_scheduler.D.params,
                  restore_step, device, ckpt_dir, adv_train,
                  training_mode,
                  config.model_G,
                  config.model_G_f0,
                  config.model_G_sp,
                  config.model_G_ap,
                  data_parallel,
                  )              

    # Loss
    loss_name = config.train.loss.name
    gamma_stft = config.train.loss.gammas_stft[0]
    gamma_feats = config.train.loss.gammas_feats[0]
    gamma_adv = config.train.loss.gammas_adv
    loss_fft_size = config.train.loss.audio.fft_size
    loss_shift_size = config.train.loss.audio.shift_size
    loss_win_length = config.train.loss.audio.win_length
    loss_type_G = config.train.loss.loss_type.G
    loss_type_D = config.train.loss.loss_type.D
    cut_edge = config.train.loss.cut_edge
    loss_step = config.train.loss.steps[0]
    i_loss = 0
    speech_synthesizer, criterion_stft, criterion_gen_adv, criterion_dis_adv = \
        get_speech_synthesizer_and_loss(
            device, sample_rate, n_fft, hop_length, synth_hop_length,
            loss_name,
            loss_fft_size, loss_shift_size, loss_win_length,
            loss_type_G, loss_type_D)

    # Validation
    lsd_cut_edge = config.val.lsd.cut_edge
    lsd_fft_size = config.val.lsd.fft_size
    lsd_shift_size = config.val.lsd.shift_size
    lsd_win_length = config.val.lsd.win_length

    # Loggers
    train_logger, val_logger, synth_logger = get_loggers(log_dir)

    # Make training faster
    torch.backends.cudnn.benchmark = config.train.cudnn.benchmark
    torch.backends.cudnn.deterministic = config.train.cudnn.deterministic

    # Steps
    train_log_step = config.train.train_log_step
    train_synth_step = config.train.train_synth_step
    train_lsd_step = config.train.train_lsd_step
    val_step = config.train.val_step
    val_lsd_step = config.train.val_step * config.train.val_lsd_n_val_step
    synth_step = config.train.val_step * config.train.synth_n_val_step

    # Epochs
    current_step = 0
    if config.train.restore_step:
        current_step = config.train.restore_step

    max_steps = config.train.max_steps
    steps_bar = tqdm(total=config.train.max_steps, desc="Steps...", position=0)

    # Training
    if training_mode == "each_feats_model":
        G_f0.train()
        G_sp.train()
        G_ap.train()
    elif training_mode == "shared_model":
        G.train()
    if adv_train:
        D.train()
    current_step = 0
    while current_step < max_steps:
        train_bar = tqdm(total=len(data_loaders["train"]), desc="Training step...", position=1)
        for batch in data_loaders["train"]:
            current_step += 1

            if training_mode == "each_feats_model":
                optimizer_G_f0.zero_grad()
                optimizer_G_sp.zero_grad()
                optimizer_G_ap.zero_grad()
            elif training_mode == "shared_model":
                optimizer_G.zero_grad()
            if adv_train:
                optimizer_D.zero_grad()

            # Device
            basenames = batch[0]
            x = batch[1].to(device)
            f0_target = batch[2].to(device)
            spectrum_target = batch[3].to(device)
            aperiodicity_target = batch[4].to(device)
            y = batch[5].to(device)

            # Loss setting
            if loss_step is not None and current_step >= loss_step:
                i_loss += 1
                gamma_stft = config.train.loss.gammas_stft[i_loss]
                gamma_feats = config.train.loss.gammas_feats[i_loss]
                if i_loss < len(config.train.loss.steps):
                    loss_step = config.train.loss.steps[i_loss]
                elif i_loss == len(config.train.loss.steps):
                    loss_step = None

            d_loss = 0
            if adv_train and current_step >= discriminate_step:
                ### Update discriminator ###
                # Discriminate target wav
                if discriminate_spec:
                    y_magspec = stft(y, 1024, 24, 1024,
                             torch.hann_window(1024).to(device)).transpose(1, 2)
                    d_out_real = D(y_magspec)
                else:
                    d_out_real = D(y.unsqueeze(1))

                # Discriminate generated wav
                if training_mode == "each_feats_model":
                    f0_output = G_f0(x)
                    spectrum_output = G_sp(x)
                    aperiodicity_output = G_ap(x)
                elif training_mode == "shared_model":
                    output_features = G(x)
                    f0_output = output_features[ : , 0 : 1, : ]
                    spectrum_output = output_features[ : , 1 : 1 + dim_sp, : ]
                    aperiodicity_output = output_features[
                        : , 1 + dim_sp : 1 + dim_sp + dim_ap, : ]
                f0_hat, spectrum_hat, aperiodicity_hat = \
                    decode_feats(
                        f0_output, spectrum_output, aperiodicity_output,
                        scaler_means, scaler_vars, sample_rate,
                        n_fft, comp, coding_ap, device)
                y_hat = speech_synthesizer(
                    f0_hat, spectrum_hat, aperiodicity_hat)
                if discriminate_spec:
                    y_hat_magspec = stft(
                        y_hat, 1024, 24, 1024,
                        torch.hann_window(1024).to(device)).transpose(1, 2)
                    d_out_gen = D(y_hat_magspec)
                else:
                    d_out_gen = D(y_hat.unsqueeze(1))

                # Loss
                d_loss_real, d_loss_fake = criterion_dis_adv(d_out_gen, d_out_real)
                d_loss = d_loss_real + d_loss_fake

                # Backward
                d_loss.backward()
                torch.nn.utils.clip_grad_norm_(D.parameters(), config.train.max_norm)
                optimizer_D.step()

            ### Update generator ###
            # Discriminate generated wav
            if training_mode == "each_feats_model":
                f0_output = G_f0(x)
                spectrum_output = G_sp(x)
                aperiodicity_output = G_ap(x)
            elif training_mode == "shared_model":
                output_features = G(x)
                if torch.isnan(output_features).sum() > 0:
                    print("model output is nan in {} (step {})".format(
                        basenames[int(torch.where(output_features.isnan())[0][0])], current_step
                    ))
                f0_output = output_features[ : , 0 : 1, : ]
                spectrum_output = output_features[ : , 1 : 1 + dim_sp, : ]
                aperiodicity_output = output_features[
                    : , 1 + dim_sp : 1 + dim_sp + dim_ap, : ]
            f0_hat, spectrum_hat, aperiodicity_hat = decode_feats(
                f0_output, spectrum_output, aperiodicity_output,
                scaler_means, scaler_vars, sample_rate, n_fft,
                comp, coding_ap, device)
            y_hat = speech_synthesizer(
                f0_hat, spectrum_hat, aperiodicity_hat)

            d_out_gen = None
            if adv_train and current_step >= discriminate_step:
                if discriminate_spec:
                    y_hat_magspec = stft(
                        y_hat, 1024, 24, 1024,
                        torch.hann_window(1024).to(device)).transpose(1, 2)
                    d_out_gen = D(y_hat_magspec)
                else:
                    d_out_gen = D(y_hat.unsqueeze(1))

            # Loss
            f0_loss = spetrum_loss = aperiodicity_loss = feature_loss = adv_loss = g_loss = stft_loss = 0
            ## STFT
            if loss_name == "stft" or loss_name == "mrstft":
                sc_loss, stft_mag_loss = criterion_stft(
                    y_hat[ : , cut_edge : y.shape[1] - cut_edge], 
                    y[ : , cut_edge : y.shape[1] - cut_edge])
                stft_loss = sc_loss + stft_mag_loss
            ## Feature
            f0_loss = F.l1_loss(f0_target.squeeze(), f0_output.squeeze())
            spetrum_loss = F.l1_loss(spectrum_target, spectrum_output)
            aperiodicity_loss = F.l1_loss(aperiodicity_target, aperiodicity_output)
            feature_loss = gamma_feats * (f0_loss + spetrum_loss + aperiodicity_loss)
            if adv_train and current_step >= discriminate_step:
                adv_loss = criterion_gen_adv(d_out_gen)
            ## Total
            g_loss = feature_loss
            if loss_name == "stft" or loss_name == "mrstft":
                if adv_train and current_step >= discriminate_step:
                    g_loss += gamma_stft * stft_loss + gamma_adv * adv_loss
                else:
                    g_loss += gamma_stft * stft_loss

            # Backward
            g_loss.backward()

            if training_mode == "each_feats_model":
                torch.nn.utils.clip_grad_norm_(G_f0.parameters(), config.train.max_norm)
                torch.nn.utils.clip_grad_norm_(G_sp.parameters(), config.train.max_norm)
                torch.nn.utils.clip_grad_norm_(G_ap.parameters(), config.train.max_norm)
                optimizer_G_f0.step()
                optimizer_G_sp.step()
                optimizer_G_ap.step()
            elif training_mode == "shared_model":
                torch.nn.utils.clip_grad_norm_(G.parameters(), config.train.max_norm)
                optimizer_G.step()

            # Train logging
            if current_step % train_log_step == 0:               
                train_logger.add_scalar("Loss/g_loss", g_loss, global_step=current_step)
                train_logger.add_scalar("Loss/stft_loss", stft_loss, global_step=current_step)
                train_logger.add_scalar("Loss/f0_loss", f0_loss, global_step=current_step)
                train_logger.add_scalar("Loss/spectrum_loss", spetrum_loss, global_step=current_step)
                train_logger.add_scalar("Loss/aperiodicity_loss", aperiodicity_loss, global_step=current_step)
                if adv_train and current_step >= discriminate_step:
                    train_logger.add_scalar("Loss/d_loss", d_loss, global_step=current_step)
                    train_logger.add_scalar("Loss/adv_loss", adv_loss, global_step=current_step)

                train_bar.write(
                    "Training loss of G: {}, Mag: {}, F0: {}, Sp: {}, Ap: {} (step {})".format(
                        g_loss, stft_loss, f0_loss, spetrum_loss, aperiodicity_loss, current_step))

            # Train synthesize
            if current_step % train_synth_step == 0:
                logging(
                    train_logger, current_step, "cpu", speech_synthesizer, scaler_means,
                    scaler_vars, basenames, x.detach().clone(),
                    f0_hat.detach().clone(), spectrum_hat.detach().clone(),
                    aperiodicity_hat.detach().clone(), y_hat.detach().clone(),
                    f0_target.detach().clone(),
                    spectrum_target.detach().clone(),
                    aperiodicity_target.detach().clone(),
                    y.detach().clone(), sample_rate, n_fft, hop_length,
                    frame_shift_ms, comp, coding_ap, n_samples=5, mode="train")

            # Train LSD
            if current_step % train_lsd_step == 0:
                if current_step // train_lsd_step == 1:
                    lsd_world_coded_pytorch, lsd_world_world, lsd_world_pytorch = \
                        calc_lsd(
                            "cpu", speech_synthesizer, scaler_means, scaler_vars,
                            sample_rate, n_fft, hop_length, frame_shift_ms,
                            lsd_fft_size, lsd_shift_size, lsd_win_length,
                            comp, coding_ap, y, f0_target=f0_target,
                            spectrum_target=spectrum_target,
                            aperiodicity_target=aperiodicity_target, 
                            lsd_cut_edge=lsd_cut_edge, mean=True,
                            use_generator=False)
                lsd_gen_world, lsd_gen_pytorch = \
                    calc_lsd(
                        "cpu", speech_synthesizer, scaler_means, scaler_vars,
                        sample_rate, n_fft, hop_length, frame_shift_ms,
                        lsd_fft_size, lsd_shift_size, lsd_win_length,
                        comp, coding_ap, y, wav_hat=y_hat, f0_hat=f0_hat,
                        spectrum_hat=spectrum_hat,
                        aperiodicity_hat=aperiodicity_hat,
                        lsd_cut_edge=lsd_cut_edge, mean=True,
                        use_generator=True)
                if current_step // train_lsd_step == 1:
                    train_logger.add_scalar("Score/lsd_world_coded_pytorch", lsd_world_coded_pytorch, global_step=current_step)
                    train_logger.add_scalar("Score/lsd_world_world", lsd_world_world, global_step=current_step)
                    train_logger.add_scalar("Score/lsd_world_pytorch", lsd_world_pytorch, global_step=current_step)
                train_logger.add_scalar("Score/lsd_gen_world", lsd_gen_world, global_step=current_step)
                train_logger.add_scalar("Score/lsd_gen_pytorch", lsd_gen_pytorch, global_step=current_step)
            # Validation
            if current_step % val_step == 0:
                g_losses = d_losses = adv_losses = stft_losses = f0_losses \
                = spectrum_losses = aperiodicity_losses = 0
                if current_step % val_lsd_step == 0:
                    lsds_gen_val = []
                    if current_step // val_lsd_step == 1:
                        lsds_wo_gen_val = []
                if training_mode == "each_feats_model":
                    G_f0.eval()
                    G_sp.eval()
                    G_ap.eval()
                elif training_mode == "shared_model":
                    G.eval()
                if adv_train and current_step >= discriminate_step:
                    D.eval()
                dev_bar = tqdm(total=len(data_loaders["dev"]), desc="Development step...", position=2)
                with torch.no_grad():
                    for batch in data_loaders["dev"]:
                        # Device
                        basenames = batch[0]
                        x = batch[1].to(device)
                        f0_target = batch[2].to(device)
                        spectrum_target = batch[3].to(device)
                        aperiodicity_target = batch[4].to(device)
                        y = batch[5].to(device)

                        if adv_train and current_step >= discriminate_step:
                            ### Validate discriminator ###
                            # Discriminate target wav
                            if discriminate_spec:
                                y_magspec = stft(
                                    y, 1024, 24, 1024,
                                    torch.hann_window(1024).to(device)
                                    ).transpose(1, 2)
                                d_out_real = D(y_magspec)
                            else:
                                d_out_real = D(y.unsqueeze(1))

                            # Discriminate generated wav
                            if training_mode == "each_feats_model":
                                f0_output = G_f0(x)
                                spectrum_output = G_sp(x)
                                aperiodicity_output = G_ap(x)
                            elif training_mode == "shared_model":
                                output_features = G(x)
                                f0_output = output_features[ : , 0 : 1, : ]
                                spectrum_output = output_features[ : , 1 : 1 + dim_sp, : ]
                                aperiodicity_output = output_features[
                                    : , 1 + dim_sp : 1 + dim_sp + dim_ap, : ]
                            f0_hat, spectrum_hat, aperiodicity_hat = \
                                decode_feats(
                                    f0_output, spectrum_output, aperiodicity_output,
                                    scaler_means, scaler_vars, sample_rate,
                                    n_fft, comp, coding_ap, device)
                            y_hat = speech_synthesizer(
                                f0_hat, spectrum_hat, aperiodicity_hat)
                            if discriminate_spec:
                                y_hat_magspec = stft(
                                    y_hat, 1024, 24, 1024,
                                    torch.hann_window(1024).to(device)).transpose(1, 2)
                                d_out_gen = D(y_hat_magspec)
                            else:
                                d_out_gen = D(y_hat.unsqueeze(1))

                            # Loss
                            d_loss_real, d_loss_fake = criterion_dis_adv(d_out_gen, d_out_real)
                            d_loss = d_loss_real + d_loss_fake

                        ### Validate generator ###
                        # Discriminate generated wav
                        if training_mode == "each_feats_model":
                            f0_output = G_f0(x)
                            spectrum_output = G_sp(x)
                            aperiodicity_output = G_ap(x)
                        elif training_mode == "shared_model":
                            output_features = G(x)
                            f0_output = output_features[ : , 0 : 1, : ]
                            spectrum_output = output_features[ : , 1 : 1 + dim_sp, : ]
                            aperiodicity_output = output_features[
                                : , 1 + dim_sp : 1 + dim_sp + dim_ap, : ]
                        f0_hat, spectrum_hat, aperiodicity_hat = \
                            decode_feats(
                                f0_output, spectrum_output, aperiodicity_output,
                                scaler_means, scaler_vars, sample_rate,
                                n_fft, comp, coding_ap, device)
                        y_hat = speech_synthesizer(
                            f0_hat, spectrum_hat, aperiodicity_hat)
                        if adv_train and current_step >= discriminate_step:
                            if discriminate_spec:
                                y_hat_magspec = stft(
                                    y_hat, 1024, 24, 1024,
                                    torch.hann_window(1024).to(device)).transpose(1, 2)
                                d_out_gen = D(y_hat_magspec)
                            else:
                                d_out_gen = D(y_hat.unsqueeze(1))

                        # Loss
                        stft_loss = f0_loss = spetrum_loss = aperiodicity_loss = feature_loss = adv_loss = g_loss = 0
                        ## STFT
                        if loss_name == "stft" or loss_name == "mrstft":
                            sc_loss, stft_mag_loss = criterion_stft(
                                y_hat[ : , cut_edge : y.shape[1] - cut_edge], 
                                y[ : , cut_edge : y.shape[1] - cut_edge])
                            stft_loss = sc_loss + stft_mag_loss
                        ## Feature
                        f0_loss = F.l1_loss(f0_target.squeeze(), f0_output.squeeze())
                        spetrum_loss = F.l1_loss(spectrum_target, spectrum_output)
                        aperiodicity_loss = F.l1_loss(aperiodicity_target, aperiodicity_output)
                        feature_loss = gamma_feats * (f0_loss + spetrum_loss + aperiodicity_loss)
                        if adv_train and current_step >= discriminate_step:
                            adv_loss = criterion_gen_adv(d_out_gen)
                        ## Total
                        g_loss = feature_loss
                        if loss_name == "stft" or loss_name == "mrstft":
                            if adv_train and current_step >= discriminate_step:
                                g_loss += gamma_stft * stft_loss + gamma_adv * adv_loss
                            else:
                                g_loss += gamma_stft * stft_loss

                        g_losses += float(g_loss)
                        d_losses += float(d_loss)
                        adv_losses += float(adv_loss)
                        stft_losses += float(stft_loss)
                        f0_losses += float(f0_loss)
                        spectrum_losses += float(spetrum_loss)
                        aperiodicity_losses += float(aperiodicity_loss)

                        # LSD
                        if current_step % val_lsd_step == 0:
                            if current_step // val_lsd_step == 1:
                                lsds_wo_gen_val.append(
                                    calc_lsd(
                                        "cpu", speech_synthesizer, scaler_means, scaler_vars,
                                        sample_rate, n_fft, hop_length, frame_shift_ms,
                                        lsd_fft_size, lsd_shift_size, lsd_win_length,
                                        comp, coding_ap, y, f0_target=f0_target,
                                        spectrum_target=spectrum_target,
                                        aperiodicity_target=aperiodicity_target, 
                                        lsd_cut_edge=lsd_cut_edge, mean=False,
                                        use_generator=False))
                            lsds_gen_val.append(
                                calc_lsd(
                                    "cpu", speech_synthesizer, scaler_means, scaler_vars,
                                    sample_rate, n_fft, hop_length, frame_shift_ms,
                                    lsd_fft_size, lsd_shift_size, lsd_win_length,
                                    comp, coding_ap, y, wav_hat=y_hat, f0_hat=f0_hat,
                                    spectrum_hat=spectrum_hat,
                                    aperiodicity_hat=aperiodicity_hat,
                                    lsd_cut_edge=lsd_cut_edge, mean=False,
                                    use_generator=True))

                        # update progress bar
                        dev_bar.update(1)

                    if current_step % synth_step == 0:
                        synth_bar = tqdm(total=len(data_loaders["synth"]), desc="Synthesize...", position=3)
                        lsds_gen_synth = []
                        if current_step // synth_step == 1:
                            lsds_wo_gen_synth = []
                        for batch in data_loaders["synth"]:
                            basename = batch[0]
                            spectrogram_all = batch[1].to(device).detach().clone()
                            f0_all = batch[2].to(device).detach().clone()
                            spectrum_all = batch[3].to(device).detach().clone()
                            aperiodicity_all = batch[4].to(device).detach().clone()
                            y_all = batch[5].to(device).detach().clone()
                            if training_mode == "each_feats_model":
                                f0_output = G_f0(spectrogram_all)
                                spectrum_output = G_sp(spectrogram_all)
                                aperiodicity_output = G_ap(spectrogram_all)
                            else:
                                output_features_all = G(spectrogram_all)
                                f0_output = output_features_all[ : , 0 : 1, : ]
                                spectrum_output = output_features_all[ : , 1 : 1 + dim_sp, : ]
                                aperiodicity_output = output_features_all[
                                    : , 1 + dim_sp : 1 + dim_sp + dim_ap, : ]
                            f0_hat, spectrum_hat, aperiodicity_hat = \
                                decode_feats(
                                    f0_output, spectrum_output,
                                    aperiodicity_output, scaler_means,
                                    scaler_vars, sample_rate, n_fft, comp,
                                    coding_ap, device)

                            y_hat_all = speech_synthesizer(
                                f0_hat, spectrum_hat, aperiodicity_hat)
                            logging(synth_logger, current_step, "cpu", speech_synthesizer,
                                    scaler_means, scaler_vars, basename,
                                    spectrogram_all, 
                                    f0_hat, spectrum_hat, aperiodicity_hat, y_hat_all,
                                    f0_all,
                                    spectrum_all, aperiodicity_all, y_all,
                                    sample_rate, n_fft, hop_length,
                                    frame_shift_ms, comp, coding_ap,
                                    n_samples=50, mode="synth")
                            if current_step // synth_step == 1:
                                lsds_wo_gen_synth.append(
                                    calc_lsd(
                                        "cpu", speech_synthesizer, scaler_means, scaler_vars,
                                        sample_rate, n_fft, hop_length, frame_shift_ms,
                                        lsd_fft_size, lsd_shift_size, lsd_win_length,
                                        comp, coding_ap, y_all, f0_target=f0_all,
                                        spectrum_target=spectrum_all,
                                        aperiodicity_target=aperiodicity_all, 
                                        lsd_cut_edge=lsd_cut_edge, mean=False,
                                        use_generator=False))
                            lsds_gen_synth.append(
                                calc_lsd(
                                    "cpu", speech_synthesizer, scaler_means, scaler_vars,
                                    sample_rate, n_fft, hop_length, frame_shift_ms,
                                    lsd_fft_size, lsd_shift_size, lsd_win_length,
                                    comp, coding_ap, y_all, wav_hat=y_hat_all, f0_hat=f0_hat,
                                    spectrum_hat=spectrum_hat,
                                    aperiodicity_hat=aperiodicity_hat,
                                    lsd_cut_edge=lsd_cut_edge, mean=False,
                                    use_generator=True))
                            synth_bar.update(1)
                        if current_step // synth_step == 1:
                            lsd_world_coded_pytorch = torch.mean(torch.cat([l[0] for l in lsds_wo_gen_synth]))
                            lsd_world_world = torch.mean(torch.cat([l[1] for l in lsds_wo_gen_synth]))
                            lsd_world_pytorch = torch.mean(torch.cat([l[2] for l in lsds_wo_gen_synth]))
                            synth_logger.add_scalar("Score/lsd_world_coded_pytorch",
                                                    lsd_world_coded_pytorch,
                                                    global_step=current_step)
                            synth_logger.add_scalar("Score/lsd_world_world",
                                                    lsd_world_world,
                                                    global_step=current_step)
                            synth_logger.add_scalar("Score/lsd_world_pytorch",
                                                    lsd_world_pytorch,
                                                    global_step=current_step)
                        lsd_gen_world = torch.mean(torch.cat([l[0] for l in lsds_gen_synth]))
                        lsd_gen_pytorch = torch.mean(torch.cat([l[1] for l in lsds_gen_synth]))
                        synth_logger.add_scalar("Score/lsd_gen_world",
                                                lsd_gen_world, 
                                                global_step=current_step)
                        synth_logger.add_scalar("Score/lsd_gen_pytorch",
                                                lsd_gen_pytorch,
                                                global_step=current_step)

                # Val logging
                n_val_step = len(data_loaders["dev"])
                if training_mode == "each_feats_model":
                    val_logger.add_scalar("Learning_rate/G", optimizer_G_f0.param_groups[0]["lr"], global_step=current_step)
                else:
                    val_logger.add_scalar("Learning_rate/G", optimizer_G.param_groups[0]["lr"], global_step=current_step)
                if adv_train:
                    val_logger.add_scalar("Learning_rate/D", optimizer_D.param_groups[0]["lr"], global_step=current_step)
                val_logger.add_scalar("Loss/g_loss", g_losses / n_val_step, global_step=current_step)
                val_logger.add_scalar("Loss/stft_loss", stft_losses / n_val_step, global_step=current_step)
                val_logger.add_scalar("Loss/f0_loss", f0_losses / n_val_step, global_step=current_step)
                val_logger.add_scalar("Loss/spectrum_loss", spectrum_losses / n_val_step, global_step=current_step)
                val_logger.add_scalar("Loss/aperiodicity_loss", aperiodicity_losses / n_val_step, global_step=current_step)
                if adv_train and current_step >= discriminate_step:
                    val_logger.add_scalar("Loss/d_loss", d_losses / n_val_step, global_step=current_step)
                    val_logger.add_scalar("Loss/adv_loss", adv_losses / n_val_step, global_step=current_step)

                dev_bar.write(
                    "Validation loss of G: {}, D: {}, Adv: {}, Mag: {}, F0: {}, Sp: {}, Ap: {} (step {})".format(
                        g_losses / n_val_step,
                        d_losses / n_val_step,
                        adv_losses / n_val_step,
                        stft_losses / n_val_step,
                        f0_losses / n_val_step,
                        spectrum_losses / n_val_step,
                        aperiodicity_losses / n_val_step,
                        current_step))

                if current_step % val_lsd_step == 0:
                    if current_step // val_lsd_step == 1:
                        lsd_world_coded_pytorch_val = torch.mean(torch.cat([l[0] for l in lsds_wo_gen_val]))
                        lsd_world_world_val = torch.mean(torch.cat([l[1] for l in lsds_wo_gen_val]))
                        lsd_world_pytorch_val = torch.mean(torch.cat([l[2] for l in lsds_wo_gen_val]))
                        val_logger.add_scalar("Score/lsd_world_coded_pytorch", lsd_world_coded_pytorch_val, global_step=current_step)
                        val_logger.add_scalar("Score/lsd_world_world", lsd_world_world_val, global_step=current_step)
                        val_logger.add_scalar("Score/lsd_world_pytorch", lsd_world_pytorch_val, global_step=current_step)
                    lsd_gen_world_val = torch.mean(torch.cat([l[0] for l in lsds_gen_val]))
                    lsd_gen_pytorch_val = torch.mean(torch.cat([l[1] for l in lsds_gen_val]))
                    val_logger.add_scalar("Score/lsd_gen_world", lsd_gen_world_val, global_step=current_step)
                    val_logger.add_scalar("Score/lsd_gen_pytorch", lsd_gen_pytorch_val, global_step=current_step)
                if training_mode == "each_feats_model":
                    G_f0.train()
                    G_sp.train()
                    G_ap.train()
                elif training_mode == "shared_model":
                    G.train()
                if adv_train and current_step >= discriminate_step:    
                    D.train()

                # Close progress bar
                dev_bar.close()

            # Save models
            if current_step % config.train.save_step == 0:
                if training_mode == "each_feats_model":
                    state_dict = {
                        "G_f0": G_f0.state_dict(),
                        "G_sp": G_sp.state_dict(),
                        "G_ap": G_ap.state_dict(),
                        "optimizer_G_f0": optimizer_G_f0.state_dict(),
                        "optimizer_G_sp": optimizer_G_sp.state_dict(),
                        "optimizer_G_ap": optimizer_G_ap.state_dict(),
                        "lr_scheduler_G_f0": lr_scheduler_G_f0.state_dict(),
                        "lr_scheduler_G_sp": lr_scheduler_G_sp.state_dict(),
                        "lr_scheduler_G_ap": lr_scheduler_G_ap.state_dict(),
                    }
                elif training_mode == "shared_model":
                    state_dict = {
                        "G": G.state_dict(),
                        "optimizer_G": optimizer_G.state_dict(),
                        "lr_scheduler_G": lr_scheduler_G.state_dict(),
                    }
                if adv_train:
                    state_dict["D"] = D.state_dict()
                    state_dict["optimizer_D"] = optimizer_D.state_dict()
                    state_dict["lr_scheduler_D"] = lr_scheduler_D.state_dict()
                torch.save(
                    state_dict, ckpt_dir / "step{}.pth.tar".format(current_step))

            # LR scheduler
            if training_mode == "each_feats_model":
                lr_scheduler_G_f0.step()
                lr_scheduler_G_sp.step()
                lr_scheduler_G_ap.step()
            elif training_mode == "shared_model":
                lr_scheduler_G.step()
            if adv_train and current_step >= discriminate_step:
                lr_scheduler_D.step()

            # update progress bar
            train_bar.update(1)
            steps_bar.update(1)     

            if current_step >= max_steps:
                break

        # Close progress bar
        train_bar.close()

    # Close progress bar
    steps_bar.close()


if __name__ == "__main__":
    train()