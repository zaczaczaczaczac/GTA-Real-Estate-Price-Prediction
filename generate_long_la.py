import pandas as pd
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut

# Initialize Geopy Geocoder with a custom user_agent
geolocator = Nominatim(user_agent="address_to_latlong")

def get_lat_lon(address):
    print(address)
    try:
        # Use geopy to geocode the address
        location = geolocator.geocode(address, timeout=10)
        if location:
            return location.latitude, location.longitude
        else:
            return None, None  # If the address cannot be geocoded
    except GeocoderTimedOut:
        return None, None  # Handle timeout errors gracefully

# Load the Excel file into a pandas DataFrame
df = pd.read_excel(path)

# Create new columns for latitude and longitude
df['Latitude'] = None
df['Longitude'] = None

# Iterate through each row in the DataFrame
for index, row in df.iterrows():
    address = row['address']
    latitude, longitude = get_lat_lon(address)
    df.at[index, 'Latitude'] = latitude
    df.at[index, 'Longitude'] = longitude

# Save the updated DataFrame back to an Excel file
output_file_path = "output_with_latlong.xlsx"  # Specify your desired output file path
df.to_excel(output_file_path, index=False)

print(f"Updated file saved as {output_file_path}")