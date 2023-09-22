#!/bin/bash

select_task () {
  if [ "$MODEL" == "DEEPLABCUT" ]
  then
    if [ "$PROJECT" == "NEW" ]
    then
      eval TASK=("create")
    else
      select_iteration
      eval TASK=( $(whiptail --title "Choose tasks" --menu \
      "Choose one or more TASK(S)" 25 78 16 \
      "modify_config" "  modify the config.yaml file of this project" \
      "create_dataset" "  create a training dataset" \
      "modify_pose_config" "  modify the pose_cfg.yaml file of this iteration" \
      "train" "  train a model" \
      "evaluate" "  evaluate a model" 3>&1 1>&2 2>&3) )
    #    "analyze_videos" "  analyze your videos" 3>&1 1>&2 2>&3) )
    fi
  elif [ "$MODEL" == "POSEC3D" ]
  then
    eval TASK=( $(whiptail --title "Choose tasks" --menu \
    "Choose one or more TASK(S)" 25 78 16 \
    "create_dataset" "  create a training dataset" \
    "train" "  train a model" 3>&1 1>&2 2>&3) )
  fi
}

select_project () {
  eval ALL_PROJECTS=( $(ls ./models/deeplabcut/))
  args_projects+=("NEW" "Create a new deeplabcut project")
  for item in "${ALL_PROJECTS[@]}"
  do
    args_projects+=("$item" "Use an existing project")
  done
  eval PROJECT=( $(whiptail --title "Chose your project" --menu "Choose an option" 25 78 16 \
  "${args_projects[@]}" 3>&1 1>&2 2>&3) )
}

select_iteration () {
  eval N_ITERATIONS=( $(ls ./models/deeplabcut/"$PROJECT"/dlc-models/ | wc))
  eval ALL_ITERATIONS=( $(ls -r ./models/deeplabcut/"$PROJECT"/dlc-models/))
  args_iterations+=("NEW" "Create a new iteration in your project")
  if [ "$N_ITERATIONS" > 0 ]
  then
    for item in "${ALL_ITERATIONS[@]}"
    do
      args_iterations+=("$item" "Use an existing iteration")
    done
  fi
  eval ITERATION=( $(whiptail --title "Chose your iteration" --menu "Choose an option" 25 78 16 \
  "${args_iterations[@]}" 3>&1 1>&2 2>&3) )
}

select_snapshot () {
  PATTERN=./models/deeplabcut/"$PROJECT"/dlc-models/"$ITERATION"/*/train/*.meta
  eval ALL_SNAPSHOTS=( $(find $PATTERN | sed -e 's/.*\(snapshot-*\)/\1/' | sed 's/\.*.meta*//' | sort -t - -k 2 -g))
  args_snapshots+=("best_test_error" "Find best snapshot*")
  args_snapshots+=("last" "Use last existing snapshot")
  for item in "${ALL_SNAPSHOTS[@]}"
    do
      args_snapshots+=("$item" "Use an existing snapshot")
    done
  eval SNAPSHOT=( $(whiptail --title "Chose your snapshot" --menu "Choose an option // *(requires prior model evaluation)" 25 78 16 \
  "${args_snapshots[@]}" 3>&1 1>&2 2>&3) )
}

select_dataset () {
  eval ALL_DATASETS=( $(ls ./data/) )
  for item in "${ALL_DATASETS[@]}"
  do
      args_datasets+=("$item" "")
  done
  eval DATASET=( $(whiptail --title "Menu example" --menu "Choose an option" 25 78 16 \
  "${args_datasets[@]}" 3>&1 1>&2 2>&3) )
}

select_mm_dataset () {
  eval ALL_MM_DATASETS=( $(ls ./models/mmaction2/data/posec3d/) )
  for item in "${ALL_MM_DATASETS[@]}"
  do
      args_mm_datasets+=("$item" "")
  done
  eval MM_DATASET=( $(whiptail --title "Menu example" --menu "Choose an option" 25 78 16 \
  "${args_mm_datasets[@]}" 3>&1 1>&2 2>&3) )
}

select_scorer () {
  SCORER=$(whiptail --inputbox "Enter scorer name:" 10 40 --title "Scorer name" "Mitch" 3>&1 1>&2 2>&3)
}

select_gpu () {
  GPU=$(whiptail --inputbox "Enter GPU id:" 10 40 --title "GPU ID" "0" 3>&1 1>&2 2>&3)
}

select_species () {
  if ! [ -f "./data/$DATASET/species.txt" ]
  then
    (grep '\"species\"'  ./data/"$DATASET"/train_annotation.json \
     | sed 's/^.*: //' | sed 's/\"\,.*$//' | cut -c 2- | sort | uniq \
     --> ./data/"$DATASET"/species.txt)
  fi
  readarray -t ALL_SPECIES < "./data/$DATASET/species.txt"
  for item in "${ALL_SPECIES[@]}"
  do
      args_species+=("$item" "" ON)
  done
  eval SPECIES=( $(whiptail --title "Choose species" --checklist \
        "Choose one or more SPECIES" 30 78 20 "${args_species[@]}" 3>&1 1>&2 2>&3) )
}

select_import_kpt () {
  eval ALL_IMPORT_KPT=( $(find ./import/*_keypoints.txt) )
  #      ALL_IMPORT_LIMB=$(ls ./import/*_limbs.txt)
  #      echo "$ALL_IMPORT_KPT"
  for item in "${ALL_IMPORT_KPT[@]}"
  do
      args_import_kpt+=("$item" "")
  done
  eval IMPORT_KPT=( $(whiptail --title "Menu example" --menu "Choose an option" 25 78 16 \
  "${args_import_kpt[@]}" 3>&1 1>&2 2>&3) )
}

select_kpts () {
  readarray -t ALL_KPTS < $IMPORT_KPT
  for item in "${ALL_KPTS[@]}"
  do
      args_kpt+=("$item" "" ON)
  done
  eval KPTS=( $(whiptail --title "Choose keypoints" --checklist \
        "Choose one or more KEYPOINTS" 30 78 20 "${args_kpt[@]}" 3>&1 1>&2 2>&3) )
}

select_visibility () {
  eval VISIBILITY=( $(whiptail --title "Choose keypoint visibility" --menu \
  "Choose if you want to use all keypoints or only the visible ones. \nKeypoints with invalid coordinates (i. e. (0, 0)) will be excluded" 25 78 16 \
  "All_keypoints" "All keypoints with valid xy-coordinates will be included" \
  "Only_visible_keypoints" "Only visible keypoints (i. e. with visibility=1) will be included " 3>&1 1>&2 2>&3) )
}

select_modification_mode () {
  eval MOD_MODE=( $(whiptail --title "Choose modification mode" --menu "Choose if you want" 25 78 16 \
  "via_GUI" "Modifies file using this GUI" \
  "via_pico" "Opens file in terminal using pico" 3>&1 1>&2 2>&3) )
}

select_shuffle () {
  eval ALL_SHUFFLES=( $(ls ./models/deeplabcut/"$PROJECT"/dlc-models/"$ITERATION"/) )
  if [ ${#ALL_SHUFFLES[@]} -gt 1 ]
  then
      args_shuffles+=("all" "use all shuffles")
    for item in "${ALL_SHUFFLES[@]}"
    do
        args_shuffles+=("$item" "")
    done
    eval SHUFFLE=( $(whiptail --title "Menu example" --menu "Choose an option" 25 78 16 \
    "${args_shuffles[@]}" 3>&1 1>&2 2>&3) )
  else
    eval SHUFFLE=${ALL_SHUFFLES[0]}
  fi
}

modify_config () {
  FILE="./models/deeplabcut/"$PROJECT"/config.yaml"
  if [ "$MOD_MODE" = "via_GUI" ]
  then
      get_modifications $FILE
      sed -i "$SED_STRING" $FILE
  elif [ "$MOD_MODE" = "via_pico" ]
  then
    (pico $FILE)
  fi
}

modify_pose_config () {
  if [ "$MOD_MODE" = "via_GUI" ]
  then
    if [ "$SHUFFLE" = "all" ]
    then
      FILE_FIRST_SHUFFLE="./models/deeplabcut/"$PROJECT"/dlc-models/"$ITERATION"/${ALL_SHUFFLES[0]}/train/pose_cfg.yaml"
      get_modifications $FILE_FIRST_SHUFFLE
      for item in "${ALL_SHUFFLES[@]}"
      do
        FILE="./models/deeplabcut/"$PROJECT"/dlc-models/"$ITERATION"/"$item"/train/pose_cfg.yaml"
        sed -i "$SED_STRING" $FILE # WARNING!! only updates files with identical parameters
      done
    else
      FILE="./models/deeplabcut/"$PROJECT"/dlc-models/"$ITERATION"/"$SHUFFLE"/train/pose_cfg.yaml"
      get_modifications $FILE
      sed -i "$SED_STRING" $FILE
    fi
  elif [ "$MOD_MODE" = "via_pico" ]
  then
    if [ "$SHUFFLE" = "all" ]
    then
      for item in "${ALL_SHUFFLES[@]}"
      do
        (pico ./models/deeplabcut/"$PROJECT"/dlc-models/"$ITERATION"/"$item"/train/pose_cfg.yaml)
      done
    else
      (pico ./models/deeplabcut/"$PROJECT"/dlc-models/"$ITERATION"/"$SHUFFLE"/train/pose_cfg.yaml)
    fi
  fi
}

get_modifications () {
  FILE=$1
  readarray -t ALL_LINES < "$FILE"
  for line in "${ALL_LINES[@]}"
  do
    if grep -q ": ." <<< $line; then
      left=$(echo "$line" | sed 's/\.*:.*//')
      right=$(echo "$line" | sed -e 's/.*: \(.*\)/\1/')
      args_mods+=("$left" "$right")
    fi
  done
  eval MOD=( $(whiptail --title "Menu example" --menu "Choose an option" 25 78 16 \
  "${args_mods[@]}" 3>&1 1>&2 2>&3) )
  for i in "${!args_mods[@]}"; do
    if [[ "${args_mods[$i]}" = "${MOD}" ]]; then
      KEY=${args_mods[$i]}
      VALUE=${args_mods[$i+1]}
        break
    fi
  done

  NEW_VALUE=$(whiptail --inputbox "Enter $KEY:" 10 40 --title "Scorer name" "$VALUE" 3>&1 1>&2 2>&3)

  SED_STRING=$"s/$KEY: $VALUE/$KEY: $NEW_VALUE/"
}

#modify_file () {
#  sed -i "s/$KEY: $VALUE/$KEY: $NEW_VALUE/" $FILE
#}

select_net_type () {
  eval NET_TYPE=( $(whiptail --title "Choose your network type" --menu "Choose an option" 25 78 16 \
  "resnet_50" "resnet_50" \
  "resnet_101" "resnet_101" \
  "resnet_152" "resnet_152" \
  "efficientnet-b0" "efficientnet-b0" \
  "efficientnet-b1" "efficientnet-b1" \
  "efficientnet-b2" "efficientnet-b2" \
  "efficientnet-b3" "efficientnet-b3" \
  "efficientnet-b4" "efficientnet-b4" \
  "efficientnet-b5" "efficientnet-b5" \
  "efficientnet-b6" "efficientnet-b6" 3>&1 1>&2 2>&3) )
}

select_cross_validation () {
  eval CROSS_VALIDATION=( $(whiptail --title "Choose cross validation" --menu "Choose an option" 25 78 16 \
  "NO" "No cross validation" \
  "5-fold-CV" "Produce a 5-fold cross validation" \
  "10-fold-CV" "Produce a 10-fold cross validation" 3>&1 1>&2 2>&3) )
}

update_training_fraction () {
  FILE="./models/deeplabcut/"$PROJECT"/config.yaml"
  if [ "$CROSS_VALIDATION" = "5-fold-CV" ]
  then
    (sed -i 'N;N;N;s/\(TrainingFraction:\n\).*\(\niteration\)/\1- 0.8\2/g' $FILE)
  elif [ "$CROSS_VALIDATION" = "10-fold-CV" ]
  then
    (sed -i 'N;N;N;s/\(TrainingFraction:\n\).*\(\niteration\)/\1- 0.9\2/g' $FILE)
  fi
}

eval MODEL=( $(whiptail --title "Choose module" --menu \
"Choose module" 20 78 2 \
"DEEPLABCUT" "For pose estimation" \
"POSEC3D" "For action recognition" 3>&1 1>&2 2>&3) )


if [ "$MODEL" = "DEEPLABCUT" ]
then
  select_project
  select_task
  if [ "$TASK" == "create" ]
  then
    select_scorer
    select_dataset
    select_species
    select_import_kpt
    select_kpts
    select_visibility
    python main.py --model $MODEL --task $TASK --scorer $SCORER --dataset $DATASET --species "${SPECIES[@]}"  --all_keypoints "${ALL_KPTS[@]}" --keypoints "${KPTS[@]}" --visibility $VISIBILITY
  elif [ "$TASK" == "modify_config" ]
  then
    select_modification_mode
    modify_config
  elif [ "$TASK" == "modify_pose_config" ]
  then
    select_modification_mode
    select_shuffle
    modify_pose_config
  elif [ "$TASK" == "create_dataset" ]
  then
    select_net_type
    select_cross_validation
    update_training_fraction
    python main.py --model $MODEL --task $TASK --project $PROJECT --iteration $ITERATION --network $NET_TYPE --cross_validation $CROSS_VALIDATION
  elif [ "$TASK" == "train" ]
  then
    select_shuffle
    select_gpu
    python main.py --model $MODEL --task $TASK --project $PROJECT --iteration $ITERATION --gpu $GPU --shuffle $SHUFFLE
#    nohup python main.py --model $MODEL --task $TASK --project $PROJECT --iteration $ITERATION --gpu $GPU --shuffle $SHUFFLE > output_train.txt &
  elif [ "$TASK" == "evaluate" ]
  then
    select_shuffle
    select_gpu
    python main.py --model $MODEL --task $TASK --project $PROJECT --iteration $ITERATION --gpu $GPU --shuffle $SHUFFLE
#    nohup python main.py --model $MODEL --task $TASK --project $PROJECT --iteration $ITERATION --gpu $GPU --shuffle $SHUFFLE > output_eval.txt &
  fi
elif [ "$MODEL" = "POSEC3D" ]
then
  select_task
  if [ "$TASK" == "create_dataset" ]
  then
    select_dataset
    select_project
    select_iteration
    select_shuffle
    select_snapshot
    select_gpu
#    nohup python main.py --model $MODEL --task $TASK --dataset $DATASET --project $PROJECT --iteration $ITERATION --snapshot $SNAPSHOT --gpu $GPU --shuffle $SHUFFLE > output_dataset.txt &
    python main.py --model $MODEL --task $TASK --dataset $DATASET --project $PROJECT --iteration $ITERATION --snapshot $SNAPSHOT --gpu $GPU --shuffle $SHUFFLE

  elif [ "$TASK" == "train" ]
  then
    select_mm_dataset
    PROJECT=$(echo $MM_DATASET | sed 's/\.*-iteration.*//')
    ITERATION=$(echo $MM_DATASET | sed -e 's/.*\(iteration-.*\)/\1/')
    python main.py --model $MODEL --task $TASK --project $PROJECT --iteration $ITERATION --mm_dataset $MM_DATASET
  fi
fi
