models=("sbert" "bert" "roberta")
datasets=("GDS" "WDC")
for data in "${datasets[@]}"
do
for i in $(seq 0.7 0.1 0.9)
do
  for p in $(seq 0.1 0.1 0.9)
  do
    for lm in "${models[@]}"
      do
        echo $i $lm $p
      /mnt/c/Users/Pierre/AppData/Local/Programs/Python/Python311/python.exe main.py --step 4 --dataset $data --similarity $i --portion $p --embed $lm --baseline
      done
    done
done
done