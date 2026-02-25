"""Remote segmentation handlers for Tiled integration."""


class RemoteSegmentationSender:
    def __init__(self):
    
        from tiled.client import from_uri

        self.client = from_uri('https://tiled.nsls2.bnl.gov')
        self.writer = self.client['tst/sandbox/synaps/reconstructions']
        self.segapp_elems = []

    def clear_cache(self):
        self.segapp_elems.clear()
    
    def append_cache(self, elem):
        self.segapp_elems.append(elem)
    
    def get_cache(self):
        return self.segapp_elems
    
    def cache_size(self):
        return len(self.segapp_elems)
    
    def write(self, data, key=None):
        """Write numpy array data to remote handler."""
        try:
            result = self.writer.write_array(data, key=key, access_tags=['synaps_project'])
            print(f"[REMOTE] Data written with key: {key}, result: {result}" if key else f"[REMOTE] Data written, result: {result}")
            return result
        except Exception as e:
            print(f"[REMOTE ERROR] Failed to write data with key {key}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def write_metadata(self, metadata_dict, key=None):
        """Write metadata as a dictionary."""
        pass


class RemoteSegmentationReceiver:
    def __init__(self, num_elements):
    
        from tiled.client import from_uri

        self.client = from_uri('https://tiled.nsls2.bnl.gov')
        self.reader = self.client['tst/sandbox/synaps/segmentations']
        self.keys = []
        self.values = []
        self.num_elements = num_elements
        self.count_connect = 0

    def subscribe(self):
        self.sub = self.reader.subscribe()
        self.sub.child_created.add_callback(self.get_keys)
        print("Listening for updates. Use Ctrl+C to stop....")
        self.sub.start()

    def get_keys(self, data):
        print(f"Received Key : {data}")
        #self.keys.append(data)
        sub = data.child().subscribe()
        sub.new_data.add_callback(self.get_data)
        sub.start_in_thread(start=1)
        #sub1.disconnect()

    def get_data(self, data):
        print(f"count num : {self.count_connect}")
        #print(f"Received Data : {data}")
        #self.values.append(data)
        self.count_connect += 1 
        if self.count_connect == self.num_elements:
            self.sub.disconnect()
