#!/bin/bash

search_links(){
    for i in $( ls -d * ); do
        
        if [ -L $i ]; then
            path_to_link=$( pwd )
            
            ls -l $i
            
            if [ ! -e $i ]; then
                pwd
                ls -l $i
                return
            fi
        fi
        
        if [[ $i == '..' ]]; then
            continue
        fi
        
        if [ -d $i ]; then
            cd $i
            #echo "MOVED INTO $i"
            search_links
            cd ..
        fi
    done
}

search_links
