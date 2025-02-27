import subprocess
from argparse import ArgumentParser
import sys
import threading
import time


def format_time(seconds):
    # 一天有86400秒（24小时 * 60分钟 * 60秒）
    days = seconds // 86400
    seconds = seconds % 86400

    # 一小时有3600秒（60分钟 * 60秒）
    hours = seconds // 3600
    seconds %= 3600

    # 一分钟有60秒
    minutes = seconds // 60
    seconds %= 60

    # 构建格式化字符串
    time_str = ""
    if days > 0:
        time_str += f"{days}天"
    if hours > 0:
        time_str += f"{hours}小时"
    if minutes > 0:
        time_str += f"{minutes}分钟"
    # 总是显示秒，即使它是0（可以根据需要调整）
    time_str += f"{seconds:.0f}秒"

    return time_str.strip()


def call(command):
    begin = time.time()
    print(f"开始执行 {' '.join(command)}")
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)

    # 定义用于读取标准输出和标准错误的线程函数
    def stream_output(stream, prefix=''):
        for line in stream:
            print(f"{prefix}{line.strip()}")

    # 定义并启动线程来读取标准输出和标准错误
    output_thread = threading.Thread(target=stream_output, args=(process.stdout, ''))
    error_thread = threading.Thread(target=stream_output, args=(process.stderr, ''))
    output_thread.start()
    error_thread.start()

    try:
        # 等待子进程结束
        process.wait()
    finally:
        # 确保关闭管道
        process.stdout.close()
        process.stderr.close()
        # 等待线程结束，确保所有输出都已处理
        output_thread.join()
        error_thread.join()

    end = time.time()
    print(f"{command[0]} {command[1]} 执行结束，返回码为：{process.returncode}，耗时：{format_time(end-begin)}")


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--tf_record', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    args = parser.parse_args()

    if args.output_dir.endswith('/') or args.output_dir.endswith('\\'):
        args.output_dir = args.output_dir[:-1]

    t1 = time.time()
    command = ['python', 'preprocess/waymo/waymo_converter.py', '--tf_record', args.tf_record, '--output_dir', args.output_dir]
    call(command)
    t2 = time.time()

    command = ['python', 'preprocess/auto_reorient.py', '--input_path', f"{args.output_dir}/camera_calibration/rectified/sparse", '--output_path', f"{args.output_dir}/camera_calibration/aligned/sparse/0"]
    call(command)
    t3 = time.time()

    command = ['python', 'preprocess/generate_chunks.py', '--project_dir', args.output_dir, '--keep_raw_chunks', '--skip_bundle_adjustment', '--chunk_size', '1000000000']
    call(command)
    t4 = time.time()

    command = ['python', 'preprocess/generate_depth.py', '--project_dir', args.output_dir]
    call(command)
    t5 = time.time()
    print(f"执行结束，总耗时：{format_time(t5-t1)}")


