#!/bin/bash

lookup_string=$1

look_for_personal_inf(){
    for i in $( ls -d * ); do

        if [[ $i == '..' ]]; then
            continue
        fi

        if [[ $i == 'venv' ]]; then
            continue
        fi
        
        if [[ $i == 'hora.txt' ]]; then
            continue
        fi

        if [[ $i == 'time.txt' ]]; then
            continue
        fi

        if [ -d $i ]; then
            cd $i
            #echo "MOVED INTO $i"
            look_for_personal_inf
            cd ..
        else
            grep -F $lookup_string $i #2>&1 | grep -ve "inputData" | grep -ve "outputData" | grep -ve ".jar" | grep -ve "Computing Generation" | grep -e "Opening the file" | grep -e "Porcentaje Aciertos"
        fi
    done
}

echo "Looking for: $lookup_string"

look_for_personal_inf

