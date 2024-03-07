# nohup sh scripts/template.sh zsclip txt                  0.7 0 >logs/out_files/clip/zsclip_txt_visda.out&
# nohup sh scripts/template.sh ft_pl_baseline txt          0.7 0 >logs/out_files/clip/ft_pl_txt_visda.out&
# nohup sh scripts/template.sh ft_pl_neg_proto_bank_v0 txt 0.7 2 >logs/out_files/clip/ft_pl_neg_proto_bank_v0_visda.out&

# nohup sh scripts/template.sh ft_pl_neg_proto_bank_v1 txt 0.7 0 >logs/out_files/clip/ft_pl_neg_proto_bank_v1.out&
# nohup sh scripts/template.sh ft_pl_neg_proto_bank_v2 txt 0.7 0 >logs/out_files/clip/ft_pl_neg_proto_bank_v2.out&
# nohup sh scripts/template.sh ft_pl_neg_proto_bank_v3 txt 0.7 0 >logs/out_files/clip/ft_pl_neg_proto_bank_v3.out&
# nohup sh scripts/template.sh ft_pl_neg_proto_bank_v3 ens_cupl 0.7 2 >logs/out_files/clip/ft_pl_neg_proto_bank_v3_ens_cupl.out&
# nohup bash scripts/template.sh ft_pl_neg_proto_bank_v4 txt 0.7 0 >logs/out_files/clip/ft_pl_neg_proto_bank_v4.out&

# nohup sh scripts/template.sh ft_pl_baseline ens_cupl     0.7 0 >logs/out_files/clip/ft_pl_ens_cupl.out
# nohup sh scripts/template.sh zsclip ens_cupl             0.7 0 >logs/out_files/clip/zsclip_ens_cupl.out


METHOD=$1
GPU_ID=$2
nohup bash scripts/template.sh ${METHOD} txt 0.7 ${GPU_ID} >logs/out_files/clip/${METHOD}.out&