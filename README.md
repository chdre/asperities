# Asperities
Extracting and shuffling voids/asperities in images. Designed for a projected where 2D images represent material structures of quartz. Shuffling allows for creating new material structures. 

# Install
Install using pip:
```
pip install git+https://github.com/chdre/asperities
```

# Usage
```
import asperities

unique_asperities = asperities.get_asperities(image)
```
Shuffle a single asperity in an image. Useful to generate new geometries.
```
shuffled_asperity = asperitiesshuffle_asperity(image)
```

Shuffle asperities in an image
```
shuffled_asperities = asperities.shuffle_asperities(image)
```
