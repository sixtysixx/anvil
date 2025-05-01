import sounddevice as sd
import sys

print("--- Querying Devices ---")
try:
    print(sd.query_devices())
except Exception as e:
    print(f"Error querying devices: {e}", file=sys.stderr)
    sys.exit(1)

print("\n--- Default Input Device ---")
try:
    input_dev_info = sd.query_devices(kind='input')
    print(input_dev_info)
    print(f"Supported input sample rates (approx): {input_dev_info.get('default_samplerate', 'N/A')}")
    # Note: query_devices doesn't list *all* supported rates easily.
    # Trying a few common rates:
    print("Testing common input rates:")
    for rate in [8000, 16000, 22050, 44100, 48000]:
        try:
            sd.check_input_settings(samplerate=rate)
            print(f"  {rate} Hz: Supported")
        except Exception as e:
            print(f"  {rate} Hz: Not Supported ({e})")

except Exception as e:
    print(f"Error querying input device: {e}", file=sys.stderr)


print("\n--- Default Output Device ---")
try:
    output_dev_info = sd.query_devices(kind='output')
    print(output_dev_info)
    print(f"Supported output sample rates (approx): {output_dev_info.get('default_samplerate', 'N/A')}")
    # Testing common/relevant rates:
    print("Testing common/relevant output rates:")
    for rate in [8000, 16000, 22050, 44100, 48000]:
        try:
            sd.check_output_settings(samplerate=rate)
            print(f"  {rate} Hz: Supported")
        except Exception as e:
            print(f"  {rate} Hz: Not Supported ({e})")

except Exception as e:
    print(f"Error querying output device: {e}", file=sys.stderr)

print("\n--- Done ---")
