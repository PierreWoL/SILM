for i in $(seq 0.1 0.1 0.9)
do
    /mnt/c/Users/Pierre/AppData/Local/Programs/Python/Python311/python.exe main.py --step 4 --dataset WDC --similarity $i
done