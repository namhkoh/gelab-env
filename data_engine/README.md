# GUI-Exploration-Lab

## Icon Resources
The `icons/` directory contains the original icon assets that will be used for UI rendering.

## Simulation Data

### Environment Generation
To generate a new simulation environment with timestamp naming in the `ui_environment/` directory, run:
```bash
python tree.py
```

### Data Structure Documentation
The `ui_structure_layer.json` file contains the following key elements:

```json
{
    "root": "Root node",
    "image": "Corresponding page image file",
    "layout": "Icon elements contained on the page",
    "bbox": "Coordinate area of the icon",
    "type": "Icon type ('normal' for regular icons, 'system' for home and back)",
    "transitions": "State transitions for the layout",
    "action": "Name of the icon being clicked",
    "target_page": "Target page of the transition",
    "subnodes": "Child nodes of the current node",
    "metadata": "Configuration used to generate the current environment"
}
```