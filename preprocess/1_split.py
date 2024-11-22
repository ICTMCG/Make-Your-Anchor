import os
import subprocess


import argparse

def get_video_duration(video_path):
    ext = os.path.splitext(video_path)[-1]
    if ext != '.mp4' and ext != '.avi' and ext != '.flv':
        print('not a video')
        return False
    ffprobe_cmd = 'ffprobe -i {} -show_entries format=duration -v quiet -of csv="p=0"'
    p = subprocess.Popen(
        ffprobe_cmd.format(video_path),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        shell=True
    )
    out, err = p.communicate()
    # something wrong
    if len(str(err, 'gbk')) > 0:
        print('our:{} err:{}'.format(out, str(err, 'gbk')))
        return False
    # video length==0
    if len(str(out, 'gbk')) == 0:
        print('video length is 0')
        return False
    second = int(float(out))
    print('video time: {}s'.format(second))
    return second


def one_func(root_dir, convert_to_30fps=False):
    video_name = 'raw.mp4'
    '''
    convert to 30 fps
    '''
    if convert_to_30fps:
        video_30fps_name = 'raw_30fps.mp4'
        cmd = 'ffmpeg -i {} -r 30 {}'.format( os.path.join(root_dir, video_name), os.path.join(root_dir, video_30fps_name) )
        os.system(cmd)
    '''
    split by 10s
    '''
    if convert_to_30fps:
        video_path = os.path.join(root_dir, video_30fps_name)
    else:
        video_path = os.path.join(root_dir, video_name)
    time = get_video_duration(video_path)
    interval=10
    for idx, start_time in enumerate( range(0, time, interval)):
        split_dir = os.path.join(root_dir, 'splits', 'split_{}'.format(idx+1))
        os.makedirs(split_dir, exist_ok=True)
        output_file = os.path.join(split_dir, 'raw.mp4' )
        end_time = start_time+interval
        cmd = f"ffmpeg -i {video_path} -ss { str(start_time) } -to { str(end_time) } -filter:v 'cropdetect=limit=100:round=2:reset=0' -c:a copy -avoid_negative_ts 1 {output_file} -y"
        os.system(cmd)
        '''
        get frames
        '''
        output_frames_dir = os.path.join(split_dir, 'image')
        os.makedirs(output_frames_dir, exist_ok=True)
        cmd = 'ffmpeg -i {}  {}/%06d.png'.format(output_file, output_frames_dir)
        os.system(cmd)
        '''
        get audio
        '''
        output_wav = os.path.join(split_dir, 'raw.wav')
        cmd = 'ffmpeg -i {}  {}'.format(output_file, output_wav)
        os.system(cmd)
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument( "--video_dir", type=str, default=None, required=True)
    parser.add_argument( "--convert_to_30fps", default=False, action='store_true')
    args = parser.parse_args()

    one_func(args.video_dir, args.convert_to_30fps)