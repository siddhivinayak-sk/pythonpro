import boto3
from botocore.exceptions import ClientError
import os

# 1. Create a method to create boto client
def create_s3_client(aws_access_key_id=None, aws_secret_access_key=None, region_name=None):
    return boto3.client(
        's3',
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        region_name=region_name
    )

# 2. Use client to upload file using Post method (presigned POST)
def upload_file_post(client, bucket_name, file_path, object_name):
    try:
        presigned_post = client.generate_presigned_post(
            Bucket=bucket_name,
            Key=object_name
        )
        with open(file_path, 'rb') as f:
            files = {'file': (object_name, f)}
            import requests
            response = requests.post(presigned_post['url'], data=presigned_post['fields'], files=files)
        return response.status_code == 204
    except Exception as e:
        print(f"POST upload error: {e}")
        return False

# 3. Use client to upload file using Put method (put_object)
def upload_file_put(client, bucket_name, file_path, object_name):
    try:
        with open(file_path, 'rb') as f:
            client.put_object(Bucket=bucket_name, Key=object_name, Body=f)
        return True
    except Exception as e:
        print(f"PUT upload error: {e}")
        return False

# 4. Use client to upload file using multi-part upload
def upload_file_multipart(client, bucket_name, file_path, object_name, part_size=5*1024*1024):
    try:
        mpu = client.create_multipart_upload(Bucket=bucket_name, Key=object_name)
        parts = []
        part_number = 1
        with open(file_path, 'rb') as f:
            while True:
                data = f.read(part_size)
                if not data:
                    break
                part = client.upload_part(
                    Bucket=bucket_name,
                    Key=object_name,
                    PartNumber=part_number,
                    UploadId=mpu['UploadId'],
                    Body=data
                )
                parts.append({'ETag': part['ETag'], 'PartNumber': part_number})
                part_number += 1
        client.complete_multipart_upload(
            Bucket=bucket_name,
            Key=object_name,
            UploadId=mpu['UploadId'],
            MultipartUpload={'Parts': parts}
        )
        return True
    except Exception as e:
        print(f"Multipart upload error: {e}")
        return False

# 5. Use client to download file
def download_file(client, bucket_name, object_name, dest_path):
    try:
        client.download_file(bucket_name, object_name, dest_path)
        return True
    except Exception as e:
        print(f"Download error: {e}")
        return False

# 6. Use client to get tags for a file
def get_file_tags(client, bucket_name, object_name):
    try:
        response = client.get_object_tagging(Bucket=bucket_name, Key=object_name)
        return response.get('TagSet', [])
    except Exception as e:
        print(f"Get tags error: {e}")
        return []

# Example usage (fill in your credentials and bucket info)
if __name__ == "__main__":
    s3_client = create_s3_client()
    # bucket = 'your-bucket-name'
    # file_path = 'local-file.txt'
    # object_name = 'uploaded-file.txt'
    # upload_file_post(s3_client, bucket, file_path, object_name)
    # upload_file_put(s3_client, bucket, file_path, object_name)
    # upload_file_multipart(s3_client, bucket, file_path, object_name)
    # download_file(s3_client, bucket, object_name, 'downloaded.txt')
    # print(get_file_tags(s3_client, bucket, object_name))
