# https://github.com/FelixYuan-YF/DiffSynth-Studio/blob/main/examples/wanvideo/README.md, about wan2.1 camera control training

# Please use the convert_csv.py first to generate metadata.csv

# ---------------------------------------------------------------
# Wan2.1-Fun-V1.1-1.3B-Control-Camera (LoRA)
# adapted form examples/wanvideo/model_training/lora/Wan2.1-Fun-V1.1-1.3B-Control-Camera.sh
# ---------------------------------------------------------------
accelerate launch examples/wanvideo/model_training/train.py \
  --dataset_base_path ./SpatialVID \
  --dataset_metadata_path metadata.csv \
  --height 480 \
  --width 832 \
  --dataset_repeat 100 \
  --model_id_with_origin_paths "PAI/Wan2.1-Fun-V1.1-1.3B-Control-Camera:diffusion_pytorch_model*.safetensors,PAI/Wan2.1-Fun-V1.1-1.3B-Control-Camera:models_t5_umt5-xxl-enc-bf16.pth,PAI/Wan2.1-Fun-V1.1-1.3B-Control-Camera:Wan2.1_VAE.pth,PAI/Wan2.1-Fun-V1.1-1.3B-Control-Camera:models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth" \
  --learning_rate 1e-5 \
  --num_epochs 5 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "./models/train/Wan2.1-Fun-V1.1-1.3B-Control-Camera_lora" \
  --lora_base_model "dit" \
  --lora_target_modules "q,k,v,o,ffn.0,ffn.2" \
  --lora_rank 32 \
  --extra_inputs "input_image,annotation_path"

# ---------------------------------------------------------------
# Wan2.1-Fun-V1.1-1.3B-Control-Camera (Full)
# adapted form examples/wanvideo/model_training/full/Wan2.1-Fun-V1.1-1.3B-Control-Camera.sh
# ---------------------------------------------------------------
accelerate launch examples/wanvideo/model_training/train.py \
  --dataset_base_path ./SpatialVID \
  --dataset_metadata_path metadata.csv \
  --height 480 \
  --width 832 \
  --dataset_repeat 100 \
  --model_id_with_origin_paths "PAI/Wan2.1-Fun-V1.1-1.3B-Control-Camera:diffusion_pytorch_model*.safetensors,PAI/Wan2.1-Fun-V1.1-1.3B-Control-Camera:models_t5_umt5-xxl-enc-bf16.pth,PAI/Wan2.1-Fun-V1.1-1.3B-Control-Camera:Wan2.1_VAE.pth,PAI/Wan2.1-Fun-V1.1-1.3B-Control-Camera:models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth" \
  --learning_rate 1e-5 \
  --num_epochs 2 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "./models/train/Wan2.1-Fun-V1.1-1.3B-Control-Camera_full" \
  --trainable_models "dit" \
  --extra_inputs "input_image,annotation_path"

# ---------------------------------------------------------------
# Wan2.1-Fun-V1.1-14B-Control-Camera (LoRA)
# adapted form examples/wanvideo/model_training/lora/Wan2.1-Fun-V1.1-14B-Control-Camera.sh
# ---------------------------------------------------------------
accelerate launch examples/wanvideo/model_training/train.py \
  --dataset_base_path ./SpatialVID \
  --dataset_metadata_path metadata.csv \
  --height 480 \
  --width 832 \
  --dataset_repeat 100 \
  --model_id_with_origin_paths "PAI/Wan2.1-Fun-V1.1-14B-Control-Camera:diffusion_pytorch_model*.safetensors,PAI/Wan2.1-Fun-V1.1-14B-Control-Camera:models_t5_umt5-xxl-enc-bf16.pth,PAI/Wan2.1-Fun-V1.1-14B-Control-Camera:Wan2.1_VAE.pth,PAI/Wan2.1-Fun-V1.1-14B-Control-Camera:models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth" \
  --learning_rate 1e-5 \
  --num_epochs 5 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "./models/train/Wan2.1-Fun-V1.1-14B-Control-Camera_lora" \
  --lora_base_model "dit" \
  --lora_target_modules "q,k,v,o,ffn.0,ffn.2" \
  --lora_rank 32 \
  --extra_inputs "input_image,annotation_path"

# ---------------------------------------------------------------
# Wan2.1-Fun-V1.1-14B-Control-Camera (Full)
# adapted form examples/wanvideo/model_training/full/Wan2.1-Fun-V1.1-14B-Control-Camera.sh
# ---------------------------------------------------------------
accelerate launch --config_file examples/wanvideo/model_training/full/accelerate_config_14B.yaml examples/wanvideo/model_training/train.py \
  --dataset_base_path ./SpatialVID \
  --dataset_metadata_path metadata.csv \
  --height 480 \
  --width 832 \
  --dataset_repeat 100 \
  --model_id_with_origin_paths "PAI/Wan2.1-Fun-V1.1-14B-Control-Camera:diffusion_pytorch_model*.safetensors,PAI/Wan2.1-Fun-V1.1-14B-Control-Camera:models_t5_umt5-xxl-enc-bf16.pth,PAI/Wan2.1-Fun-V1.1-14B-Control-Camera:Wan2.1_VAE.pth,PAI/Wan2.1-Fun-V1.1-14B-Control-Camera:models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth" \
  --learning_rate 1e-5 \
  --num_epochs 2 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "./models/train/Wan2.1-Fun-V1.1-14B-Control-Camera_full" \
  --trainable_models "dit" \
  --extra_inputs "input_image,annotation_path"
