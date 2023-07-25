from datetime import datetime, timedelta
import whisperx
import os
import pandas as pd
import argparse


class Video2SRT:

    def __init__(self):
        self.device = "cuda"
        self.compute_type = "float16"

    def get_model(self):
        whisper_model = whisperx.load_model("large-v2", self.device, compute_type=self.compute_type)
        return whisper_model

    def asr(self, input_video, whisper_model):
        audio_path = self.convertvideo2audio(input_video)
        device = "cuda"
        audio_file = audio_path
        batch_size = 16  # reduce if low on GPU mem

        audio = whisperx.load_audio(audio_file)
        result = whisper_model.transcribe(audio, batch_size=batch_size)

        # 2. Align whisper output
        model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
        result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)

        # print(result["segments"]) # after alignment
        data = result["segments"]
        return data

    def convertvideo2audio(self, input_video):
        sr = 16000
        audio_path = "output.wav"
        query = f'ffmpeg -i "{input_video}" -ac 1 -acodec pcm_s16le -ar {sr} "{audio_path}" -y'
        os.system(query)
        return audio_path

    def seconds_to_timestamp(self, seconds_with_milliseconds):
        # Extract seconds and milliseconds
        seconds = int(seconds_with_milliseconds)
        milliseconds = int((seconds_with_milliseconds - seconds) * 1000)

        # Create a timedelta object with the given seconds and milliseconds
        delta = timedelta(seconds=seconds, milliseconds=milliseconds)

        # Get the timestamp by adding the delta to a reference time (midnight)
        reference_time = datetime.strptime('00:00:00', '%H:%M:%S')
        timestamp = reference_time + delta

        # Format the timestamp as "00:00:35,151"
        formatted_timestamp = timestamp.strftime('%H:%M:%S,%f')[:-3]

        return formatted_timestamp

    def dataframe_to_srt(self, dataframe, output_srt_file):
        with open(output_srt_file, 'w') as f:
            for index, row in dataframe.iterrows():
                # Convert time in seconds to SRT format (hh:mm:ss,sss)
                start_time = row['StartTime']
                end_time = row['EndTime']

                # Write subtitle entry to the SRT file
                f.write(str(index + 1) + '\n')
                f.write(start_time + ' --> ' + end_time + '\n')
                f.write(row['Text'] + '\n')
                f.write('\n')

    def video2srt(self, input_video):
        output_srt_file = 'output.srt'
        df = pd.DataFrame(columns=["Text", "StartTime", "EndTime"])
        whisper_model = self.get_model()
        asr_data = self.asr(input_video, whisper_model)
        for tmp in asr_data:
            tp = [tmp['text'], self.seconds_to_timestamp(tmp['start']), self.seconds_to_timestamp(tmp['end'])]
            df_length = len(df)
            df.loc[df_length] = tp

        df.to_csv("asrResults.csv")
        self.dataframe_to_srt(df, output_srt_file)


if __name__ == '__main__':
    my_parser = argparse.ArgumentParser()
    my_parser.add_argument('--file_path', action='store', type=str, required=True)
    args = my_parser.parse_args()
    file_path = args.file_path
    obj = Video2SRT()
    obj.video2srt(file_path)





