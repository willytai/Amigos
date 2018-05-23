# participant 12 21 22 23 24 22 are not available

for participant in $(seq 1 11)
do
    for video in $(seq 1 16)
    do 
        python3 EEG_extract.py ${participant} ${video}
    done
done


for participant in $(seq 13 20)
do
    for video in $(seq 1 16)
    do 
        python3 EEG_extract.py ${participant} ${video}
    done
done


for participant in $(seq 25 32)
do
    for video in $(seq 1 16)
    do 
        python3 EEG_extract.py ${participant} ${video}
    done
done


for participant in $(seq 34 40)
do
    for video in $(seq 1 16)
    do 
        python3 EEG_extract.py ${participant} ${video}
    done
done
