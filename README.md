# Asperities
Extracting and shuffling voids/asperities in images. Designed for a projected where 2D images represent material structures with voids. Shuffling allows for creating new material structures. 

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

# Examples
Shuffle a single asperity in an image. Useful to generate new geometries.
```
shuffled_asperity = asperities.shuffle_asperity(image)
```

Shuffle multiple asperities in an image:
```
shuffled_asperities = asperities.shuffle_asperities(image)
```
