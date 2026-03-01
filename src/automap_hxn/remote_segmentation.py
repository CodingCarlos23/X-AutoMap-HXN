"""Remote segmentation handlers for Tiled integration."""

from tiled.client.stream import LiveTableData
import threading


class RemoteSegmentationSender:
    def __init__(self, tiled_client):
        self.client = tiled_client
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
            result = self.client.write_array(data, key=key, access_tags=['synaps_project'])
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
    def __init__(self, tiled_client, num_tables):
        self.client = tiled_client
        self.num_expected = self._num_left = num_tables
        self.METADATA_UPDATES = {}
        self.results = {}
        self._lock = threading.Event()
        self._subs = []

    def wait_for_results(self):
        """Block until all expected tables are received."""
        print(f"Waiting for {self.num_expected} table{'s' if self.num_expected != 1 else ''} to be received...")
        self._lock.wait()  # blocks here until self._lock.set() is called
        print("All expected tables received. Disconnecting subscription...")
        for sub in self._subs:
            sub.disconnect()
        self.sub.disconnect()

        return self.results

    def subscribe(self):
        self.sub = self.client.subscribe()
        self.sub.child_created.add_callback(self.get_dataset)
        print("Listening for updates. Use Ctrl+C to stop....")
        self.sub.start_in_thread()

    def get_dataset(self, update):
        print(f"New dataset created: `{update.key}`. Waiting for tables to be uploaded...")
        path_parts = tuple(update.subscription.segments)
        self.METADATA_UPDATES[path_parts] = update
        sub = update.child().subscribe()
        sub.child_created.add_callback(self.get_table)
        sub.start_in_thread(start=0)
        self._subs.append(sub)

    def get_table(self, update):
        print(f"New table created: `{update.key}`. Waiting for data to be uploaded...")
        sub = update.child().subscribe()
        sub.new_data.add_callback(self.get_data)
        sub.start_in_thread(start=0)
        self._subs.append(sub)

    def get_data(self, update: LiveTableData):
        path_parts = tuple(update.subscription.segments)
        channel = path_parts[-1]  # Key is the table name
        print(f"Received data for table: `{channel}`")
        self.results[channel] = update.data()

        # If all expected tables have been received, unblock the main thread
        self._num_left -= 1
        if not self._num_left:
            self._lock.set()
