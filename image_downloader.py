from bing_image_downloader import downloader

query_string = 'Virat Kohli Face only' 
output_dir = 'dataset'

def download_images(query_string, putput_dir):
    downloader.download(query_string , limit=100,  output_dir=output_dir, adult_filter_off=True, force_replace=False, timeout=60, verbose=True)