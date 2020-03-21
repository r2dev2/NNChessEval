#/bin/bash
while :
do
	python3 Model.py -t
	mv model.pickle softmaxtanh002rate.pickle
	sleep 60
done
