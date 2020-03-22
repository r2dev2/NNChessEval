#/bin/bash
while :
do
	python3 Model.py -t
	python3 Tester.py model.pt -t >> testresults.log
	sleep 60
done
