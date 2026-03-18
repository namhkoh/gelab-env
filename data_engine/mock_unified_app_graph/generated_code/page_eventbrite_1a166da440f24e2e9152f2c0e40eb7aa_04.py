# page_id: page_eventbrite_1a166da440f24e2e9152f2c0e40eb7aa_04
# screenshot: 2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-6.png
# step_index: 4/16
# task: Open Eventbrite. Check "Sports" category. Filter events happening next month. Add the first event to your wishlist.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw background and structural UI elements for the Filters screen.
# Available variables: canvas (PIL Image), draw (PIL ImageDraw), font_sm, font_md, font_lg, font_xl

W, H = canvas.size

# Colors
status_bar_color = (236, 236, 236)        # light gray status bar
divider_light = (240, 240, 245)           # very light divider
divider = (220, 220, 225)                 # subtle section divider
card_border = (170, 170, 180)             # border for cards/buttons
card_fill = (255, 255, 255)               # white card fill (keeps contrast)
muted_bg = (250, 250, 252)                # faint page background tint
shadow_strip = (245, 245, 248)            # thin shadow strip above sticky footer

# 1) Page background (keep white/near-white to match screenshot)
draw.rectangle([0, 0, W, H], fill=card_fill)

# 2) Status bar area (top ~72px)
status_h = 72
draw.rectangle([0, 0, W, status_h], fill=status_bar_color)

# 3) Header divider below the toolbar/title area
# Header title is around y ~116; draw a subtle divider slightly below it.
header_div_y = 180
draw.line([(40, header_div_y), (W-40, header_div_y)], fill=divider_light, width=2)

# 4) Section separators between major filter groups.
# Use the detected label Y positions offset slightly so we don't draw over pasted text.
section_label_positions = {
    "categories_end_text": 1153,   # "Show less categories"
    "event_type": 1464,           # "Event type"
    "languages": 1910,            # "Languages"
    "price": 2249,                # "Price"
    "sort_by": 2567               # "Sort by"
}
# Draw thin separators a little above each section's label baseline
for name, y in section_label_positions.items():
    sep_y = max(status_h + 8, y - 20)
    # draw across most of the width with horizontal padding
    draw.line([(36, sep_y), (W-36, sep_y)], fill=divider, width=1)

# 5) Background panel for the "Sort by" control area (light rounded panel behind tabs)
sort_panel_top = 2480
sort_panel_bottom = 2740
panel_margin_x = 36
panel_radius = 14
# rounded rectangle (subtle fill) behind the sort controls
try:
    draw.rounded_rectangle(
        [panel_margin_x, sort_panel_top, W - panel_margin_x, sort_panel_bottom],
        radius=panel_radius,
        fill=muted_bg,
        outline=divider_light,
        width=1
    )
except Exception:
    # fallback if rounded_rectangle is not available
    draw.rectangle([panel_margin_x, sort_panel_top, W - panel_margin_x, sort_panel_bottom], fill=muted_bg, outline=divider_light)

# 6) Sticky "Apply filters" button background (rounded rect with border and subtle shadow)
apply_x, apply_y = 48, 2768
apply_w, apply_h = 1344, 144
apply_box = [apply_x, apply_y, apply_x + apply_w, apply_y + apply_h]
apply_radius = 12

# Thin shadow strip above the sticky area
draw.rectangle([apply_x, apply_y-12, apply_x + apply_w, apply_y-4], fill=shadow_strip)

# Button/card background and border
try:
    draw.rounded_rectangle(apply_box, radius=apply_radius, fill=card_fill, outline=card_border, width=3)
except Exception:
    draw.rectangle(apply_box, fill=card_fill, outline=card_border, width=3)

# 7) Very subtle left/right edge padding guides (light vertical lines) to structure content columns
# These are faint and extend through the content area, not overlapping status/header.
pad_x = 36
draw.line([(pad_x, header_div_y+8), (pad_x, apply_y-20)], fill=divider_light, width=1)
draw.line([(W - pad_x, header_div_y+8), (W - pad_x, apply_y-20)], fill=divider_light, width=1)

# 8) Small accent top-of-footer divider to separate content from sticky footer
footer_div_y = apply_y - 24
draw.line([(36, footer_div_y), (W-36, footer_div_y)], fill=divider, width=1)

# 9) Subtle large content area shading for long scroll region (very light)
content_top = header_div_y + 12
content_bottom = apply_y - 40
try:
    draw.rectangle([40, content_top, W-40, content_bottom], fill=None, outline=None)
except Exception:
    pass

# End of structural drawing.
# (Do not draw any icons, text, or controls that correspond to detected elements.)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_04_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-6/00_icon_Food_Drink.png
try:
    _c0 = get_crop(0, 312, 144)
    canvas.paste(_c0, (512, 383), _c0)
except Exception:
    pass
layout["Food_&_Drink"] = [512, 383, 824, 527]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_04_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-6/01_icon_Music.png
try:
    _c1 = get_crop(1, 187, 127)
    canvas.paste(_c1, (36, 383), _c1)
except Exception:
    pass
layout["Music"] = [36, 383, 223, 510]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_04_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-6/02_icon_Health.png
try:
    _c2 = get_crop(2, 199, 144)
    canvas.paste(_c2, (777, 510), _c2)
except Exception:
    pass
layout["Health"] = [777, 510, 976, 654]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_04_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-6/03_icon_Community.png
try:
    _c3 = get_crop(3, 294, 144)
    canvas.paste(_c3, (848, 383), _c3)
except Exception:
    pass
layout["Community"] = [848, 383, 1142, 527]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_04_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-6/04_icon_Business.png
try:
    _c4 = get_crop(4, 241, 144)
    canvas.paste(_c4, (247, 383), _c4)
except Exception:
    pass
layout["Business"] = [247, 383, 488, 527]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_04_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-6/05_icon_Government.png
try:
    _c5 = get_crop(5, 310, 144)
    canvas.paste(_c5, (734, 764), _c5)
except Exception:
    pass
layout["Government"] = [734, 764, 1044, 908]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_04_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-6/06_icon_Expo.png
try:
    _c6 = get_crop(6, 167, 144)
    canvas.paste(_c6, (614, 1464), _c6)
except Exception:
    pass
layout["Expo"] = [614, 1464, 781, 1608]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_04_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-6/07_icon_Auto_Boat_Air.png
try:
    _c7 = get_crop(7, 369, 144)
    canvas.paste(_c7, (449, 891), _c7)
except Exception:
    pass
layout["Auto,_Boat_&_Air"] = [449, 891, 818, 1035]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_04_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-6/08_icon_Holiday.png
try:
    _c8 = get_crop(8, 218, 127)
    canvas.paste(_c8, (492, 764), _c8)
except Exception:
    pass
layout["Holiday"] = [492, 764, 710, 891]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_04_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-6/09_icon_Spirituality.png
try:
    _c9 = get_crop(9, 282, 144)
    canvas.paste(_c9, (870, 637), _c9)
except Exception:
    pass
layout["Spirituality"] = [870, 637, 1152, 781]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_04_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-6/10_icon_Arts.png
try:
    _c10 = get_crop(10, 152, 127)
    canvas.paste(_c10, (1166, 383), _c10)
except Exception:
    pass
layout["Arts"] = [1166, 383, 1318, 510]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_04_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-6/11_icon_Spanish.png
try:
    _c11 = get_crop(11, 225, 144)
    canvas.paste(_c11, (519, 1910), _c11)
except Exception:
    pass
layout["Spanish"] = [519, 1910, 744, 2054]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_04_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-6/12_icon_Fashion.png
try:
    _c12 = get_crop(12, 220, 144)
    canvas.paste(_c12, (1068, 764), _c12)
except Exception:
    pass
layout["Fashion"] = [1068, 764, 1288, 908]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_04_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-6/13_icon_French.png
try:
    _c13 = get_crop(13, 205, 144)
    canvas.paste(_c13, (768, 1910), _c13)
except Exception:
    pass
layout["French"] = [768, 1910, 973, 2054]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_04_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-6/14_icon_Seminar.png
try:
    _c14 = get_crop(14, 232, 144)
    canvas.paste(_c14, (358, 1464), _c14)
except Exception:
    pass
layout["Seminar"] = [358, 1464, 590, 1608]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_04_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-6/15_icon_Film_Media.png
try:
    _c15 = get_crop(15, 315, 127)
    canvas.paste(_c15, (36, 510), _c15)
except Exception:
    pass
layout["Film_&_Media"] = [36, 510, 351, 637]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_04_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-6/16_icon_Italian.png
try:
    _c16 = get_crop(16, 191, 144)
    canvas.paste(_c16, (997, 1910), _c16)
except Exception:
    pass
layout["Italian"] = [997, 1910, 1188, 2054]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_04_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-6/17_icon_Family_Education.png
try:
    _c17 = get_crop(17, 432, 144)
    canvas.paste(_c17, (36, 764), _c17)
except Exception:
    pass
layout["Family_&_Education"] = [36, 764, 468, 908]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_04_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-6/18_icon_Convention.png
try:
    _c18 = get_crop(18, 293, 144)
    canvas.paste(_c18, (805, 1464), _c18)
except Exception:
    pass
layout["Convention"] = [805, 1464, 1098, 1608]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_04_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-6/19_icon_Science_Tech.png
try:
    _c19 = get_crop(19, 361, 144)
    canvas.paste(_c19, (1000, 510), _c19)
except Exception:
    pass
layout["Science_&_Tech"] = [1000, 510, 1361, 654]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_04_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-6/20_icon_Sports_Fitness.png
try:
    _c20 = get_crop(20, 378, 144)
    canvas.paste(_c20, (375, 510), _c20)
except Exception:
    pass
layout["Sports_&_Fitness"] = [375, 510, 753, 654]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_04_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-6/21_icon_Home_Lifestyle.png
try:
    _c21 = get_crop(21, 389, 127)
    canvas.paste(_c21, (36, 891), _c21)
except Exception:
    pass
layout["Home_&_Lifestyle"] = [36, 891, 425, 1018]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_04_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-6/22_icon_Charity.png
try:
    _c22 = get_crop(22, 397, 144)
    canvas.paste(_c22, (449, 637), _c22)
except Exception:
    pass
layout["Charity"] = [449, 637, 846, 781]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_04_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-6/23_icon_Festival.png
try:
    _c23 = get_crop(23, 219, 144)
    canvas.paste(_c23, (1122, 1464), _c23)
except Exception:
    pass
layout["Festival"] = [1122, 1464, 1341, 1608]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_04_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-6/24_icon_Hobbies.png
try:
    _c24 = get_crop(24, 231, 144)
    canvas.paste(_c24, (842, 891), _c24)
except Exception:
    pass
layout["Hobbies"] = [842, 891, 1073, 1035]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_04_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-6/25_icon_German.png
try:
    _c25 = get_crop(25, 225, 135)
    canvas.paste(_c25, (270, 1910), _c25)
except Exception:
    pass
layout["German"] = [270, 1910, 495, 2045]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_04_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-6/26_icon_English.png
try:
    _c26 = get_crop(26, 210, 135)
    canvas.paste(_c26, (36, 1910), _c26)
except Exception:
    pass
layout["English"] = [36, 1910, 246, 2045]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_04_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-6/27_icon_Travel_Outdoor.png
try:
    _c27 = get_crop(27, 389, 127)
    canvas.paste(_c27, (36, 637), _c27)
except Exception:
    pass
layout["Travel_&_Outdoor"] = [36, 637, 425, 764]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_04_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-6/28_icon_Conference.png
try:
    _c28 = get_crop(28, 298, 135)
    canvas.paste(_c28, (36, 1464), _c28)
except Exception:
    pass
layout["Conference"] = [36, 1464, 334, 1599]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_04_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-6/29_icon_School_Activities.png
try:
    _c29 = get_crop(29, 392, 135)
    canvas.paste(_c29, (36, 1018), _c29)
except Exception:
    pass
layout["School_Activities"] = [36, 1018, 428, 1153]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_04_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-6/30_icon_Apply_filters.png
try:
    _c30 = get_crop(30, 1344, 144)
    canvas.paste(_c30, (48, 2768), _c30)
except Exception:
    pass
layout["Apply_filters"] = [48, 2768, 1392, 2912]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_04_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-6/31_icon_5.31.png
try:
    _c31 = get_crop(31, 144, 144)
    canvas.paste(_c31, (12, 72), _c31)
except Exception:
    pass
layout["5.31"] = [12, 72, 156, 216]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_04_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-6/32_icon_5.31.png
try:
    _c32 = get_crop(32, 66, 64)
    canvas.paste(_c32, (110, 0), _c32)
except Exception:
    pass
layout["5.31"] = [110, 0, 176, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_04_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-6/33_icon_5.31.png
try:
    _c33 = get_crop(33, 61, 63)
    canvas.paste(_c33, (180, 0), _c33)
except Exception:
    pass
layout["5.31"] = [180, 0, 241, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_04_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-6/34_icon_icon_34.png
try:
    _c34 = get_crop(34, 64, 61)
    canvas.paste(_c34, (308, 2), _c34)
except Exception:
    pass
layout["icon_34"] = [308, 2, 372, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_04_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-6/35_icon_Clear_all.png
try:
    _c35 = get_crop(35, 55, 66)
    canvas.paste(_c35, (1319, 0), _c35)
except Exception:
    pass
layout["Clear_all"] = [1319, 0, 1374, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_04_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-6/36_icon_Clear_all.png
try:
    _c36 = get_crop(36, 101, 64)
    canvas.paste(_c36, (1211, 0), _c36)
except Exception:
    pass
layout["Clear_all"] = [1211, 0, 1312, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_04_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-6/37_icon_icon_37.png
try:
    _c37 = get_crop(37, 51, 62)
    canvas.paste(_c37, (248, 1), _c37)
except Exception:
    pass
layout["icon_37"] = [248, 1, 299, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_04_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-6/38_icon_clickable_35.png
try:
    _c38 = get_crop(38, 144, 144)
    canvas.paste(_c38, (1248, 2364), _c38)
except Exception:
    pass
layout["clickable_35"] = [1248, 2364, 1392, 2508]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_04_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-6/39_icon_Clear_all.png
try:
    _c39 = get_crop(39, 178, 144)
    canvas.paste(_c39, (1214, 72), _c39)
except Exception:
    pass
layout["Clear_all"] = [1214, 72, 1392, 216]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_04_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-6/40_icon_5.31.png
try:
    _c40 = get_crop(40, 102, 64)
    canvas.paste(_c40, (7, 1), _c40)
except Exception:
    pass
layout["5.31"] = [7, 1, 109, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_04_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-6/41_text_Filters.png
try:
    _c41 = get_crop(41, 180, 66)
    canvas.paste(_c41, (631, 116), _c41)
except Exception:
    pass
layout["Filters"] = [631, 116, 811, 182]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_04_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-6/42_text_Categories.png
try:
    _c42 = get_crop(42, 187, 127)
    canvas.paste(_c42, (36, 383), _c42)
except Exception:
    pass
layout["Categories"] = [36, 383, 223, 510]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_04_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-6/43_text_Show_less_categories.png
try:
    _c43 = get_crop(43, 550, 144)
    canvas.paste(_c43, (0, 1153), _c43)
except Exception:
    pass
layout["Show_less_categories"] = [0, 1153, 550, 1297]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_04_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-6/44_text_Event_type.png
try:
    _c44 = get_crop(44, 298, 135)
    canvas.paste(_c44, (36, 1464), _c44)
except Exception:
    pass
layout["Event_type"] = [36, 1464, 334, 1599]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_04_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-6/45_text_Show_all_event_types.png
try:
    _c45 = get_crop(45, 535, 144)
    canvas.paste(_c45, (0, 1599), _c45)
except Exception:
    pass
layout["Show_all_event_types"] = [0, 1599, 535, 1743]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_04_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-6/46_text_Languages.png
try:
    _c46 = get_crop(46, 210, 135)
    canvas.paste(_c46, (36, 1910), _c46)
except Exception:
    pass
layout["Languages"] = [36, 1910, 246, 2045]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_04_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-6/47_text_Show_all_languages.png
try:
    _c47 = get_crop(47, 511, 144)
    canvas.paste(_c47, (0, 2045), _c47)
except Exception:
    pass
layout["Show_all_languages"] = [0, 2045, 511, 2189]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_04_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-6/48_text_Price.png
try:
    _c48 = get_crop(48, 149, 63)
    canvas.paste(_c48, (45, 2249), _c48)
except Exception:
    pass
layout["Price"] = [45, 2249, 194, 2312]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_04_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-6/49_text_Only_free_events.png
try:
    _c49 = get_crop(49, 511, 144)
    canvas.paste(_c49, (0, 2045), _c49)
except Exception:
    pass
layout["Only_free_events"] = [0, 2045, 511, 2189]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_04_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-6/50_text_Sort_by.png
try:
    _c50 = get_crop(50, 206, 75)
    canvas.paste(_c50, (42, 2567), _c50)
except Exception:
    pass
layout["Sort_by"] = [42, 2567, 248, 2642]
