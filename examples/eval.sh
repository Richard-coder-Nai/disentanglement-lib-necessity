export OMP_NUM_THREADS=1
for vae in beta_vae
do
  for dir in 3dshapes_vae_checkpoints
  do
    python evaluate_dci.py --aff_type MI_factor_sum \
      --ckpt_path $(ls -d ../$dir/$vae/*/ | head -3) \
      --rotate 1 \
       1>../$dir/$vae/rotate.log 2>&1 &
    echo $dir/$vae
  done
  wait
done