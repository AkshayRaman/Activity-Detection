__author__ = "Akshay Raman"
import glob
import gzip
import os
import re
import sys
import shutil

class DataPreProcessor:
    ''' '''
    def __init__(self):
        ''' '''
        self.gz_file_list = []
        self.timestamp_hash = {}
        self.dir = ''
        if len(sys.argv) < 2:
            print("Folder not specified!")
            sys.exit(1)
        self.curr_dir = os.getcwd()
        self.dir = sys.argv[1]
        os.chdir(self.dir)

        #indices
        self.labels = ["walking", "standing", "sitting", "laying_down"]
        self.shorthand = {j:str(i) for i,j in enumerate(self.labels)}
        self.error = "9"

        self.ext = ".csv"

        #Each group has the same resolution, which is why they're in one group
        #They're expected to have "similar" number of timestamps...
        self.groups = {
                "sensor_group1" : ["linear_acceleration", "gravity", "accelerometer"],
                "sensor_group2" : ["gyroscope", "rotation_vector", "orientation"]
            }

        #Change the below line to use only specific sensors
        #self.groups["all_sensors"] = self.groups["sensor_group1"] + self.groups["sensor_group2"]
        self.groups["only_accelerometer"] = ["accelerometer"]
        del self.groups["sensor_group1"]
        del self.groups["sensor_group2"]

        self.file_handles = {}

    def get_gz_files(self):
        ''' '''
        self.gz_file_list = sorted(glob.glob("[0-9]*/data/*android.sensor*.csv.gz"))
        if not self.gz_file_list:
            print("No data files found!")
            sys.exit(1)

    def read_gz_file(self, gz_file, index=None):
        ''' '''
        if index:
            print("%s/%s" %(index, len(self.gz_file_list)))
        print("Processing: %s" % gz_file)
        raw_gz_file = os.path.basename(gz_file)
        headers = []

        new_file_name = raw_gz_file
        new_file_name = new_file_name.replace(".gz", "")
        new_file_name = new_file_name.replace(".data", "")
        junk_re = re.compile("^.*android\.sensor\.", re.I)
        new_file_name = junk_re.sub("", new_file_name)

        case_1 = set(["accelerometer.csv",
                "magnetic_field.csv",
                "gyroscope.csv",
                "gravity.csv",
                "linear_acceleration.csv"])
        case_2 = set(["rotation_vector.csv"])
        case_3 = set(["step_counter.csv"])
        case_4 = set(["orientation.csv"])
        case_5 = set(["light.csv"])

        if new_file_name in case_1:
            headers = ["x", "y", "z"]
        elif new_file_name in case_2:
            headers = ["a", "b", "c", "d", "e"]
        elif new_file_name in case_3:
            headers = ["steps"]
        elif new_file_name in case_4:
            headers = ["z", "x", "y"]
        elif new_file_name in case_5:
            headers = ["lux", "x1", "x2"]
        else:
            print("Invalid file: %s" % gz_file)
            return

        headers = ["timestamp"] + headers + ["reporting_mode", "label"]
        try:
            with gzip.open(gz_file, 'rb') as f:
                for line in f:
                    raw_line = line
                    line = line.strip().split(',')
                    '''
                    if len(line) != len(headers):
                        print(gz_file, line, headers)
                        continue
                    '''
                    #Get rid of reporting mode
                    del line[-2]

                    #move label to second slot
                    label = line[-1]
                    label_id = self.shorthand.get(label, self.error)
                    if label_id == self.error:
                        #print("Invalid label: '%s' in file %s" %(label, gz_file))
                        pass
                    line.insert(1, label_id)
                    line.insert(2, "O")
                    del line[-1]

                    key = "%s_%s" %(label, new_file_name)
                    fp = self.file_handles.get(key)
                    if not fp:
                        fp = open(key, 'a')
                        self.file_handles[key] = fp
                    fp.write('%s\n' % ','.join(line))

        except IOError:
            print("ERROR in gz file: %s" % gz_file)

    def close_file_handles(self):
        ''' '''
        for k in self.file_handles:
            self.file_handles[k].close()

    def cleanup_old_files(self):
        ''' '''
        file_list = glob.glob("*%s" %(self.ext))
        print("Deleting old files: %s" % file_list)
        for f in file_list:
            #Do not delete final output file
            if not f.startswith(self.dir):
                os.remove(f)

    def create_new_line(self, line1, line2, ts1, new_ts, ts2):
        ''' '''
        fraction = (1.*(new_ts-ts1))/(1.*(ts2-ts1))
        line1 = line1.split(',')
        line2 = line2.split(',')
        #'A' indicates that it has been added
        new_line = [str(new_ts), line1[1], "A"]
        for i in range(len(new_line),len(line1)):
            v1, v2 = float(line1[i]), float(line2[i])
            new_v = v1 + fraction * (v2 - v1)
            new_line.append(str(new_v))
        return ','.join(new_line)

    def fix_missing_timestamps(self):
        '''
        3 passes
        Pass 1: Find all missing timestamps
        Pass 2: Write all missing timestamps, if possible
        Pass 3: Delete timestamps that aren't present in all files
        '''

        for label in self.labels:
            for g in self.groups:
                file_group = self.groups[g]
                timestamp_sensor_hash = {}
                #pass 1
                for sensor in file_group:
                    raw_name = "%s_%s%s" %(label, sensor, self.ext)
                    if not os.path.exists(raw_name):
                        print("ERROR: File: %s does not exist" % raw_name)
                        continue

                    with open(raw_name, 'r') as fp:
                        for line in fp:
                            line = line.strip().split(',')
                            ts = int(line[0])
                            timestamp_sensor_hash[ts] = timestamp_sensor_hash.get(ts, 0) + 1

                del_keys = set()
                for k in timestamp_sensor_hash:
                    v = timestamp_sensor_hash[k]
                    if v == len(file_group):
                        del_keys.add(k)

                for k in del_keys:
                    del timestamp_sensor_hash[k]

                missing_ts = sorted(timestamp_sensor_hash.keys())
                if len(missing_ts)==0:
                    continue

                #pass 2
                for sensor in file_group:
                    add_count = 0
                    c = 0
                    new_ts = missing_ts[c]
                    raw_name = "%s_%s%s" %(label, sensor, self.ext)
                    new_name = "new_%s" % raw_name
                    if not os.path.exists(raw_name):
                        print("ERROR: File: %s does not exist" % raw_name)
                        continue

                    with open(raw_name, 'r') as fp, open(new_name, 'w') as new_fp:
                        line1 = fp.readline()
                        ts1 = int(line1.split(',')[0])
                        new_fp.write(line1)

                        #Can add data only if it is between two data points...
                        while new_ts < ts1 and c < len(missing_ts)-1:
                            c += 1
                            new_ts = missing_ts[c]

                        while True:
                            line2 = fp.readline()
                            if not line2:
                                break

                            ts2 = int(line2.split(',')[0])
                            while ts1 < new_ts and new_ts < ts2:
                                new_line = self.create_new_line(line1, line2, ts1, new_ts, ts2)
                                timestamp_sensor_hash[new_ts] = timestamp_sensor_hash.get(new_ts, 0) + 1
                                new_fp.write("%s\n" % new_line)
                                add_count += 1

                                c += 1
                                if c >= len(missing_ts):
                                    break
                                new_ts = missing_ts[c]

                            new_fp.write(line2)
                            line1,ts1 = line2,ts2

                    print("Added %s/%s timestamps to %s" %(add_count, len(missing_ts), raw_name))
                    shutil.move(new_name, raw_name)

                del_timestamps = set()
                for k in timestamp_sensor_hash:
                    v = timestamp_sensor_hash[k]
                    if v != len(file_group):
                        del_timestamps.add(k)

                if del_timestamps:
                    #pass 3
                    for sensor in file_group:
                        del_count = 0
                        raw_name = "%s_%s%s" %(label, sensor, self.ext)
                        new_name = "new_%s" % raw_name
                        if not os.path.exists(raw_name):
                            print("ERROR: File: %s does not exist" % raw_name)
                            continue

                        with open(raw_name, 'r') as fp, open(new_name, 'w') as new_fp:
                            for line1 in fp:
                                ts1 = int(line1.split(',')[0])
                                if ts1 in del_timestamps:
                                    del_count += 1
                                    continue
                                new_fp.write(line1)
                        print("Deleted %s lines from %s" %(del_count, raw_name))
                        shutil.move(new_name, raw_name)


    def combine_files(self):
        ''' '''
        FIRST_CMD = 'cut -f -2,4- -d, %s'
        NEXT_CMD = "cut -f 4- -d, %s"
        PASTE_CMD = 'paste -d, %s'
        shell_script = "run.sh"
        fp = open(shell_script, 'w')
        for label in self.labels:
            for g in self.groups:
                file_groups = self.groups[g]
                fg = []
                for i,j in enumerate(file_groups):
                    _f = "%s_%s%s" %(label, j, self.ext)
                    if not os.path.exists(_f):
                        print("Error: File %s doesn't exist!" % _f)
                        continue

                    if i==0:
                        fg.append(FIRST_CMD %_f)
                    else:
                        fg.append(NEXT_CMD % _f)

                if len(fg) != len(file_groups):
                    print("No data for %s" % label)
                    continue

                fg = [" <(%s)" % i for i in fg]
                cmd  = PASTE_CMD % ' '.join(fg)
                cmd = "%s >> %s" % (cmd, "%s_%s%s" %(self.dir, g, self.ext))
                fp.write('%s\n' % cmd)

        fp.close()
        os.system("bash %s" % shell_script)
        os.remove(shell_script)

    def finish(self):
        ''' '''
        os.chdir(self.curr_dir)

    def run(self):
        ''' '''
        self.cleanup_old_files()
        self.get_gz_files()
        for i, _file in enumerate(self.gz_file_list):
            self.read_gz_file(_file, index=i)
        self.close_file_handles()
        self.fix_missing_timestamps()
        self.combine_files()
        print("DONE!")

if __name__ == "__main__":
    dpp = DataPreProcessor()
    dpp.run()

