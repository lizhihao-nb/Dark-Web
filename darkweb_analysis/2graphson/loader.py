import subprocess
import sys
import time
import multiprocessing
import os
import re
import file_property
import jpype
from jpype.types import *

def loading():
    jar_paths = [
        '/opt/',
        '/public/home/blockchain_2/slave1/bulk_loading/lib/*',
        '/public/home/blockchain_2/slave1/bulk_load-2.0-release.jar'
    ]
    # classpath = ":".join(jar_paths)
    if not jpype.isJVMStarted():
        jpype.startJVM(classpath=jar_paths)
    BulkLoadClass = jpype.JClass("bulk_loading.bulk_load")
    args = JArray(JString)([f"/public/home/blockchain_2/slave1/darkanalysis/darkweb.properties"])
    BulkLoadClass.main(args)
    # print(result)
    jpype.shutdownJVM()
    print("finished")

if __name__ == "__main__":
    # check whether to load 
    print("start waiting!")
    file_path = f"/public/home/blockchain_2/slave1/darkanalysis/darknet_graphson"
    for i in range(6):
        filename = f"Dark{i:05d}.json"
        print(time.time())
        file = os.path.join(file_path, filename)
        janus_property_file = f"/public/home/blockchain_2/slave1/darkanalysis/darkweb.properties"
        janus_content = file_property.parse(janus_property_file)
        janus_content.put("gremlin.hadoop.inputLocation", file)
        process = multiprocessing.Process(target=loading)
        process.start()
        process.join()

    print(time.time())
