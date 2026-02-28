"""Remote segmentation handlers for Tiled integration."""


class RemoteSegmentationSender:
    def __init__(self):
    
        from tiled.client import from_uri

        self.client = from_uri('https://tiled.nsls2.bnl.gov')
        self.writer = self.client['tst/sandbox/eugene/synaps/reconstructions']
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
        self.num_elements = num_elements
        self.count_connect = 0
        self.METADATA_UPDATES = {}
        self.data_w_metadata = []

    def subscribe(self):
        self.sub = self.reader.subscribe()
        self.sub.child_created.add_callback(self.get_keys)
        print("Listening for updates. Use Ctrl+C to stop....")
        self.sub.start()

    def get_keys(self, data):
        print(f"Received Key : {data}")
        path_parts = tuple(data.subscription.segments)  # e.g. ('tst', 'sandbox', ...)
        self.METADATA_UPDATES[path_parts] = data
        sub = data.child().subscribe()
        sub.new_data.add_callback(self.get_data)
        sub.start_in_thread(start=1)

    def get_data(self, update):
        print(f"Received data number : {self.count_connect}")
        data = update.data()  # Extract the numpy array from the update.
        # Look up the metadata which we should have already received.
        path_parts = tuple(update.subscription.segments)  # e.g. ('tst', 'sandbox', ...)
        update = METADATA_UPDATES.pop(path_parts)
        metadata = update.metadata
        self.data_w_metadata.append((metadata, data))
        self.count_connect += 1 
        if self.count_connect == self.num_elements:
            self.sub.disconnect()

