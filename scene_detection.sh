for filename in D:/scenedetect_test/input_video3/*.mp4; do
	scenedetect --input $filename -d content -t 30 --csv-output  "D:/scenedetect_test/output_result2/$(basename "$filename" .mp4).csv"
done
