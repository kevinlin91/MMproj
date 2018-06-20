for filename in ./user_video/*.mp4; do
	scenedetect --input $filename -d content -t 30 --csv-output  "./scene_detection/$(basename "$filename" .mp4).csv"
done
