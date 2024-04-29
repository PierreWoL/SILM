models=("sbert" "bert" "roberta")
for i in $(seq 0.7 0.1 0.9)
do
  for p in $(seq 0.1 0.1 0.9)
  do
    for lm in "${models[@]}"
      do
        echo $i $lm $p
      /mnt/c/Users/Pierre/AppData/Local/Programs/Python/Python311/python.exe main.py --step 4 --dataset GDS --similarity $i --portion $p --embed $lm --baseline
      done
    done
done