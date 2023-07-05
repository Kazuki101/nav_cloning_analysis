for i in `seq 10`
do
  roslaunch nav_cloning_analysis nav_cloning_pytorch.launch mode:=use_dl_output
  sleep 10
done
