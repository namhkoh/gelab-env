# page_id: page_eventbrite_39adaf730c584c5582b89d1335e0c2cd_03
# screenshot: 2024_4_23_19_42_39adaf730c584c5582b89d1335e0c2cd-5.png
# step_index: 3/6
# task: Open Eventbrite. Search for 'food and drink' events. Follow the organizer of the first event in listing.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw UI background and structural elements for the provided canvas.
# Available variables:
# - canvas: PIL Image (1440x2960 RGB, white)
# - draw: PIL ImageDraw
# - font_sm, font_md, font_lg, font_xl

W, H = canvas.size

# Colors
status_bar_color = (200, 200, 200)       # light grey status bar
header_bg = (255, 255, 255)              # white header (keeps icons/text pasted on top readable)
header_underline = (33, 100, 255)        # vivid blue underline for search bar
divider_color = (230, 230, 230)          # very light grey separators
card_bg = (250, 251, 253)                # subtle off-white card backgrounds
bottom_nav_bg = (255, 255, 255)          # white bottom nav
subtle_shadow = (240, 240, 240)          # slight shadow lines

# 1) Status bar (top area ~64px tall)
status_h = 64
draw.rectangle([(0, 0), (W, status_h)], fill=status_bar_color)

# Add a faint bottom line/shadow under status bar to separate from header
draw.line([(0, status_h), (W, status_h)], fill=subtle_shadow, width=1)

# 2) Header / Search area background (below status bar)
# Reserve area for search field and underline. Keep background white.
header_top = status_h
header_bottom = 260  # ample area to include search field and underline region
draw.rectangle([(0, header_top), (W, header_bottom)], fill=header_bg)

# Slight divider above the underline to add depth
draw.line([(0, header_bottom - 56), (W, header_bottom - 56)], fill=subtle_shadow, width=1)

# 3) Blue underline for the search field (thin horizontal accent)
underline_left = 48
underline_right = W - 48
underline_y = header_bottom - 20
underline_height = 6
draw.rectangle([(underline_left, underline_y),
                (underline_right, underline_y + underline_height)],
               fill=header_underline)

# 4) Main content area background (subtle off-white band behind the list)
content_top = header_bottom + 24
content_bottom = H - 200  # leave space for bottom nav
draw.rectangle([(0, content_top), (W, content_bottom)], fill=(255,255,255))  # keep white base

# 5) Subtle rounded card backgrounds for each event row (keep very light)
# Positions derived from detected element y coordinates to align visually with pasted items.
event_rows = [
    (48, 330, W - 48, 330 + 240),   # first event block area
    (48, 720, W - 48, 720 + 240),   # second
    (48, 1110, W - 48, 1110 + 240), # third
    (48, 1500, W - 48, 1500 + 240), # fourth
    (48, 1890, W - 48, 1890 + 240), # fifth
]
for (l, t, r, b) in event_rows:
    # Slightly inset rounded card to group the content visually; very subtle fill
    draw.rounded_rectangle([(l, t), (r, b)], radius=8, fill=card_bg, outline=None)

# 6) Thin separators between list items (light grey lines)
separator_x0 = 48
separator_x1 = W - 48
separators_y = [330 + 240 + 10, 720 + 240 + 10, 1110 + 240 + 10, 1500 + 240 + 10, 1890 + 240 + 10]
for y in separators_y:
    # a faint 1px divider
    draw.line([(separator_x0, y), (separator_x1, y)], fill=divider_color, width=1)

# 7) Accent left edge line for the list area (very faint)
draw.line([(40, content_top), (40, content_bottom)], fill=subtle_shadow, width=1)

# 8) Bottom navigation bar background and top divider
bottom_nav_top = 2804  # as indicated by detected elements
bottom_nav_bottom = H
draw.rectangle([(0, bottom_nav_top), (W, bottom_nav_bottom)], fill=bottom_nav_bg)
# top divider line of bottom nav
draw.line([(0, bottom_nav_top), (W, bottom_nav_top)], fill=divider_color, width=2)

# 9) Small soft shadow above bottom nav for separation
shadow_y = bottom_nav_top - 6
draw.line([(0, shadow_y), (W, shadow_y)], fill=subtle_shadow, width=1)

# 10) Subtle left/right padding guides (non-intrusive, very light) to align content visually
pad_x = 48
draw.line([(pad_x, header_bottom + 8), (pad_x, bottom_nav_top - 8)], fill=(245,245,245), width=1)
draw.line([(W - pad_x, header_bottom + 8), (W - pad_x, bottom_nav_top - 8)], fill=(245,245,245), width=1)

# Done - structural elements and backgrounds drawn. Icons/text will be pasted on top at their positions.

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/39adaf730c584c5582b89d1335e0c2cd/step_03_2024_4_23_19_42_39adaf730c584c5582b89d1335e0c2cd-5/00_icon_SUNDAY_23RD_JUN.png
try:
    _c0 = get_crop(0, 1344, 396)
    canvas.paste(_c0, (48, 786), _c0)
except Exception:
    pass
layout["SUNDAY_23RD_JUN"] = [48, 786, 1392, 1182]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/39adaf730c584c5582b89d1335e0c2cd/step_03_2024_4_23_19_42_39adaf730c584c5582b89d1335e0c2cd-5/01_icon_Mon.png
try:
    _c1 = get_crop(1, 1344, 396)
    canvas.paste(_c1, (48, 390), _c1)
except Exception:
    pass
layout["Mon,"] = [48, 390, 1392, 786]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/39adaf730c584c5582b89d1335e0c2cd/step_03_2024_4_23_19_42_39adaf730c584c5582b89d1335e0c2cd-5/02_icon_TING.png
try:
    _c2 = get_crop(2, 1344, 396)
    canvas.paste(_c2, (48, 1182), _c2)
except Exception:
    pass
layout["@TING"] = [48, 1182, 1392, 1578]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/39adaf730c584c5582b89d1335e0c2cd/step_03_2024_4_23_19_42_39adaf730c584c5582b89d1335e0c2cd-5/03_icon_Food_and_Drink.png
try:
    _c3 = get_crop(3, 1344, 191)
    canvas.paste(_c3, (48, 72), _c3)
except Exception:
    pass
layout["Food_and_Drink"] = [48, 72, 1392, 263]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/39adaf730c584c5582b89d1335e0c2cd/step_03_2024_4_23_19_42_39adaf730c584c5582b89d1335e0c2cd-5/04_icon_Sun.png
try:
    _c4 = get_crop(4, 1344, 396)
    canvas.paste(_c4, (48, 1974), _c4)
except Exception:
    pass
layout["Sun,"] = [48, 1974, 1392, 2370]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/39adaf730c584c5582b89d1335e0c2cd/step_03_2024_4_23_19_42_39adaf730c584c5582b89d1335e0c2cd-5/05_icon_icon_5.png
try:
    _c5 = get_crop(5, 54, 62)
    canvas.paste(_c5, (314, 3), _c5)
except Exception:
    pass
layout["icon_5"] = [314, 3, 368, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/39adaf730c584c5582b89d1335e0c2cd/step_03_2024_4_23_19_42_39adaf730c584c5582b89d1335e0c2cd-5/06_icon_7.44.png
try:
    _c6 = get_crop(6, 53, 63)
    canvas.paste(_c6, (183, 2), _c6)
except Exception:
    pass
layout["7.44"] = [183, 2, 236, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/39adaf730c584c5582b89d1335e0c2cd/step_03_2024_4_23_19_42_39adaf730c584c5582b89d1335e0c2cd-5/07_icon_7.44.png
try:
    _c7 = get_crop(7, 58, 65)
    canvas.paste(_c7, (114, 1), _c7)
except Exception:
    pass
layout["7.44"] = [114, 1, 172, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/39adaf730c584c5582b89d1335e0c2cd/step_03_2024_4_23_19_42_39adaf730c584c5582b89d1335e0c2cd-5/08_icon_icon_8.png
try:
    _c8 = get_crop(8, 45, 60)
    canvas.paste(_c8, (252, 4), _c8)
except Exception:
    pass
layout["icon_8"] = [252, 4, 297, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/39adaf730c584c5582b89d1335e0c2cd/step_03_2024_4_23_19_42_39adaf730c584c5582b89d1335e0c2cd-5/09_icon_MC.png
try:
    _c9 = get_crop(9, 1344, 396)
    canvas.paste(_c9, (48, 1578), _c9)
except Exception:
    pass
layout["MC"] = [48, 1578, 1392, 1974]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/39adaf730c584c5582b89d1335e0c2cd/step_03_2024_4_23_19_42_39adaf730c584c5582b89d1335e0c2cd-5/10_icon_Cancel.png
try:
    _c10 = get_crop(10, 90, 65)
    canvas.paste(_c10, (1215, 0), _c10)
except Exception:
    pass
layout["Cancel"] = [1215, 0, 1305, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/39adaf730c584c5582b89d1335e0c2cd/step_03_2024_4_23_19_42_39adaf730c584c5582b89d1335e0c2cd-5/11_icon_7.44.png
try:
    _c11 = get_crop(11, 119, 104)
    canvas.paste(_c11, (54, 118), _c11)
except Exception:
    pass
layout["7.44"] = [54, 118, 173, 222]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/39adaf730c584c5582b89d1335e0c2cd/step_03_2024_4_23_19_42_39adaf730c584c5582b89d1335e0c2cd-5/12_icon_7.44.png
try:
    _c12 = get_crop(12, 90, 62)
    canvas.paste(_c12, (17, 2), _c12)
except Exception:
    pass
layout["7.44"] = [17, 2, 107, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/39adaf730c584c5582b89d1335e0c2cd/step_03_2024_4_23_19_42_39adaf730c584c5582b89d1335e0c2cd-5/13_icon_Tickets.png
try:
    _c13 = get_crop(13, 288, 156)
    canvas.paste(_c13, (864, 2804), _c13)
except Exception:
    pass
layout["Tickets"] = [864, 2804, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/39adaf730c584c5582b89d1335e0c2cd/step_03_2024_4_23_19_42_39adaf730c584c5582b89d1335e0c2cd-5/14_icon_Best_of_British_Food_and_Drink_Market.png
try:
    _c14 = get_crop(14, 1344, 396)
    canvas.paste(_c14, (48, 1974), _c14)
except Exception:
    pass
layout["Best_of_British_Food_and_"] = [48, 1974, 1392, 2370]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/39adaf730c584c5582b89d1335e0c2cd/step_03_2024_4_23_19_42_39adaf730c584c5582b89d1335e0c2cd-5/15_icon_Cancel.png
try:
    _c15 = get_crop(15, 50, 63)
    canvas.paste(_c15, (1321, 1), _c15)
except Exception:
    pass
layout["Cancel"] = [1321, 1, 1371, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/39adaf730c584c5582b89d1335e0c2cd/step_03_2024_4_23_19_42_39adaf730c584c5582b89d1335e0c2cd-5/16_icon_Cancel.png
try:
    _c16 = get_crop(16, 144, 144)
    canvas.paste(_c16, (1099, 96), _c16)
except Exception:
    pass
layout["Cancel"] = [1099, 96, 1243, 240]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/39adaf730c584c5582b89d1335e0c2cd/step_03_2024_4_23_19_42_39adaf730c584c5582b89d1335e0c2cd-5/17_icon_8_111_creator_followers.png
try:
    _c17 = get_crop(17, 1344, 396)
    canvas.paste(_c17, (48, 1974), _c17)
except Exception:
    pass
layout["8_111_creator_followers"] = [48, 1974, 1392, 2370]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/39adaf730c584c5582b89d1335e0c2cd/step_03_2024_4_23_19_42_39adaf730c584c5582b89d1335e0c2cd-5/18_icon_Wymondham_Food_and_Drink_Festival.png
try:
    _c18 = get_crop(18, 1344, 396)
    canvas.paste(_c18, (48, 786), _c18)
except Exception:
    pass
layout["Wymondham_Food_and_Drink_"] = [48, 786, 1392, 1182]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/39adaf730c584c5582b89d1335e0c2cd/step_03_2024_4_23_19_42_39adaf730c584c5582b89d1335e0c2cd-5/19_icon_Wymondham.png
try:
    _c19 = get_crop(19, 1344, 396)
    canvas.paste(_c19, (48, 786), _c19)
except Exception:
    pass
layout["Wymondham"] = [48, 786, 1392, 1182]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/39adaf730c584c5582b89d1335e0c2cd/step_03_2024_4_23_19_42_39adaf730c584c5582b89d1335e0c2cd-5/20_icon_Cancel.png
try:
    _c20 = get_crop(20, 149, 144)
    canvas.paste(_c20, (1243, 97), _c20)
except Exception:
    pass
layout["Cancel"] = [1243, 97, 1392, 241]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/39adaf730c584c5582b89d1335e0c2cd/step_03_2024_4_23_19_42_39adaf730c584c5582b89d1335e0c2cd-5/21_icon_Search_events.png
try:
    _c21 = get_crop(21, 288, 156)
    canvas.paste(_c21, (288, 2804), _c21)
except Exception:
    pass
layout["Search_events"] = [288, 2804, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/39adaf730c584c5582b89d1335e0c2cd/step_03_2024_4_23_19_42_39adaf730c584c5582b89d1335e0c2cd-5/22_icon_Home.png
try:
    _c22 = get_crop(22, 288, 156)
    canvas.paste(_c22, (0, 2804), _c22)
except Exception:
    pass
layout["Home"] = [0, 2804, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/39adaf730c584c5582b89d1335e0c2cd/step_03_2024_4_23_19_42_39adaf730c584c5582b89d1335e0c2cd-5/23_icon_8518_creator_followers.png
try:
    _c23 = get_crop(23, 1344, 396)
    canvas.paste(_c23, (48, 1578), _c23)
except Exception:
    pass
layout["8518_creator_followers"] = [48, 1578, 1392, 1974]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/39adaf730c584c5582b89d1335e0c2cd/step_03_2024_4_23_19_42_39adaf730c584c5582b89d1335e0c2cd-5/24_icon_Food_and_drinks.png
try:
    _c24 = get_crop(24, 1344, 396)
    canvas.paste(_c24, (48, 390), _c24)
except Exception:
    pass
layout["Food_and_drinks"] = [48, 390, 1392, 786]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/39adaf730c584c5582b89d1335e0c2cd/step_03_2024_4_23_19_42_39adaf730c584c5582b89d1335e0c2cd-5/25_icon_The_delicious_Quiz_of_food_and_drink.png
try:
    _c25 = get_crop(25, 1344, 396)
    canvas.paste(_c25, (48, 1182), _c25)
except Exception:
    pass
layout["The_delicious_Quiz_of_foo"] = [48, 1182, 1392, 1578]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/39adaf730c584c5582b89d1335e0c2cd/step_03_2024_4_23_19_42_39adaf730c584c5582b89d1335e0c2cd-5/26_icon_Food_and_Drink.png
try:
    _c26 = get_crop(26, 46, 63)
    canvas.paste(_c26, (384, 2), _c26)
except Exception:
    pass
layout["Food_and_Drink"] = [384, 2, 430, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/39adaf730c584c5582b89d1335e0c2cd/step_03_2024_4_23_19_42_39adaf730c584c5582b89d1335e0c2cd-5/27_icon_Favorites.png
try:
    _c27 = get_crop(27, 288, 156)
    canvas.paste(_c27, (576, 2804), _c27)
except Exception:
    pass
layout["Favorites"] = [576, 2804, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/39adaf730c584c5582b89d1335e0c2cd/step_03_2024_4_23_19_42_39adaf730c584c5582b89d1335e0c2cd-5/28_icon_More.png
try:
    _c28 = get_crop(28, 288, 156)
    canvas.paste(_c28, (1152, 2804), _c28)
except Exception:
    pass
layout["More"] = [1152, 2804, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/39adaf730c584c5582b89d1335e0c2cd/step_03_2024_4_23_19_42_39adaf730c584c5582b89d1335e0c2cd-5/29_icon_High_Spirits_Cocktail_Company.png
try:
    _c29 = get_crop(29, 1344, 396)
    canvas.paste(_c29, (48, 1182), _c29)
except Exception:
    pass
layout["High_Spirits_Cocktail_Com"] = [48, 1182, 1392, 1578]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/39adaf730c584c5582b89d1335e0c2cd/step_03_2024_4_23_19_42_39adaf730c584c5582b89d1335e0c2cd-5/30_text_Events.png
try:
    _c30 = get_crop(30, 186, 56)
    canvas.paste(_c30, (46, 301), _c30)
except Exception:
    pass
layout["Events"] = [46, 301, 232, 357]
