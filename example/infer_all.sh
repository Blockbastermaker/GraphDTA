
MODEL_DIR="/data1/zlzzheng/drugai/deepaff_data/Ki"

for t in `cat target_list`; do 

  python ../inference.py -i csv_files/${t}.csv -f 3fam_fasta/ -d 3fam_data/$t/test \
         -o results/results_$t -e results/performance_$t -m $MODEL_DIR

done
