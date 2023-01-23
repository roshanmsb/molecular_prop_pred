#! /bin/sh

for i in "bbbp" "bace_classification" "ecoli" "aa_746" "aa_1625" "impens" "mpro" "schilling" "delaney" "lipo" "freesolv" "sider" "tox21" "toxcast" "clintox" "qm7" "qm8" "qm9"
do
for j in "FG"
do
for k in {0..4}
do
python finetune.py --config conf/finetune.yaml --model.method $j --model.task_name $i --data.fold_index $k --data.task_name $i --trainer.logger.class_path lightning.pytorch.loggers.WandbLogger --trainer.logger.init_args.name pretrain_$j --trainer.logger.init_args.project $i
done
done
done
