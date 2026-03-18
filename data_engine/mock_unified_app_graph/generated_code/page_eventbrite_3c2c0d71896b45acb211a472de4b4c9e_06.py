# page_id: page_eventbrite_3c2c0d71896b45acb211a472de4b4c9e_06
# screenshot: 2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-8.png
# step_index: 6/15
# task: Open Eventbrite. Search free Health event in Los Angeles. Select the first one that is not promoted. Follow the organizer. Share to google keep notes.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Canvas and draw are provided. Draw the background and structural UI elements only.

# Colors
bg_color = (250, 251, 253)           # very light off-white background
status_color = (244, 246, 248)       # light status bar background
toolbar_color = (255, 255, 255)      # toolbar/search area (white)
muted_divider = (230, 232, 236)      # faint divider lines
accent_blue = (48, 84, 255)          # bright accent blue for underline
card_bg = (249, 250, 255)            # very subtle bluish card background

W, H = canvas.size

# Overall background
draw.rectangle((0, 0, W, H), fill=bg_color)

# Top status bar
status_h = 84
draw.rectangle((0, 0, W, status_h), fill=status_color)
# subtle bottom line under status bar
draw.line((0, status_h, W, status_h), fill=muted_divider, width=1)

# Toolbar / search area below status bar
toolbar_top = status_h
toolbar_bottom = 260
draw.rectangle((0, toolbar_top, W, toolbar_bottom), fill=toolbar_color)
# subtle shadow line under toolbar
draw.line((0, toolbar_bottom, W, toolbar_bottom), fill=muted_divider, width=1)

# Search underline (accent) — spans with left/right padding matching UI margins
underline_x0 = 48
underline_x1 = W - 48
underline_y = toolbar_bottom - 8
draw.line((underline_x0, underline_y, underline_x1, underline_y), fill=accent_blue, width=4)

# "Pills" / filter area background (subtle rounded card behind Nearby / Online groups)
pills_top = toolbar_bottom + 40
pills_bottom = pills_top + 200
pills_left = 32
pills_right = W - 32
draw.rounded_rectangle((pills_left, pills_top, pills_right, pills_bottom),
                       radius=16, fill=card_bg, outline=None)

# Divider under the pills area
divider_y = pills_bottom + 20
draw.line((32, divider_y, W - 32, divider_y), fill=muted_divider, width=1)

# "Found locations" header area background (very subtle)
found_top = divider_y + 24
found_bottom = found_top + 120
draw.rectangle((0, found_top, W, found_bottom), fill=toolbar_color)

# Thin separators between major sections (keep faint)
draw.line((32, found_bottom, W - 32, found_bottom), fill=muted_divider, width=1)

# Light separators to hint groupings for the long list (do not draw text or icons)
# Use detected item Y positions to place faint separators between major rows.
list_item_ys = [840, 1020, 1200, 1380, 1560, 1740, 1920, 2100, 2280, 2460]
for y in list_item_ys:
    # draw an extremely faint line slightly above each item's top to separate items visually
    sep_y = y - 24
    # keep it very soft so pasted text/icons remain primary
    draw.line((48, sep_y, W - 48, sep_y), fill=(240, 241, 243), width=1)

# Bottom safe area hint (very subtle)
bottom_hint_top = H - 120
draw.rectangle((0, bottom_hint_top, W, H), fill=bg_color)

# small accent left margin guide (non-intrusive)
draw.line((48, found_top - 8, 48, H - 48), fill=(246, 247, 249), width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_06_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-8/00_icon_icon_0.png
try:
    _c0 = get_crop(0, 46, 68)
    canvas.paste(_c0, (1155, 0), _c0)
except Exception:
    pass
layout["icon_0"] = [1155, 0, 1201, 68]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_06_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-8/01_icon_icon_1.png
try:
    _c1 = get_crop(1, 97, 65)
    canvas.paste(_c1, (1215, 0), _c1)
except Exception:
    pass
layout["icon_1"] = [1215, 0, 1312, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_06_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-8/02_icon_9.41.png
try:
    _c2 = get_crop(2, 168, 168)
    canvas.paste(_c2, (0, 72), _c2)
except Exception:
    pass
layout["9.41"] = [0, 72, 168, 240]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_06_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-8/03_icon_9.41.png
try:
    _c3 = get_crop(3, 54, 64)
    canvas.paste(_c3, (114, 1), _c3)
except Exception:
    pass
layout["9.41"] = [114, 1, 168, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_06_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-8/04_icon_9.41.png
try:
    _c4 = get_crop(4, 59, 63)
    canvas.paste(_c4, (178, 1), _c4)
except Exception:
    pass
layout["9.41"] = [178, 1, 237, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_06_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-8/05_icon_icon_5.png
try:
    _c5 = get_crop(5, 85, 95)
    canvas.paste(_c5, (1310, 286), _c5)
except Exception:
    pass
layout["icon_5"] = [1310, 286, 1395, 381]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_06_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-8/06_icon_icon_6.png
try:
    _c6 = get_crop(6, 50, 62)
    canvas.paste(_c6, (1320, 1), _c6)
except Exception:
    pass
layout["icon_6"] = [1320, 1, 1370, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_06_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-8/07_icon_San_Francisco.png
try:
    _c7 = get_crop(7, 1440, 132)
    canvas.paste(_c7, (0, 840), _c7)
except Exception:
    pass
layout["San_Francisco"] = [0, 840, 1440, 972]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_06_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-8/08_icon_District_of_Columbia.png
try:
    _c8 = get_crop(8, 1440, 132)
    canvas.paste(_c8, (0, 1740), _c8)
except Exception:
    pass
layout["District_of_Columbia"] = [0, 1740, 1440, 1872]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_06_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-8/09_icon_icon_9.png
try:
    _c9 = get_crop(9, 53, 63)
    canvas.paste(_c9, (315, 1), _c9)
except Exception:
    pass
layout["icon_9"] = [315, 1, 368, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_06_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-8/10_icon_Chicago.png
try:
    _c10 = get_crop(10, 1440, 132)
    canvas.paste(_c10, (0, 1380), _c10)
except Exception:
    pass
layout["Chicago"] = [0, 1380, 1440, 1512]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_06_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-8/11_icon_icon_11.png
try:
    _c11 = get_crop(11, 55, 62)
    canvas.paste(_c11, (247, 2), _c11)
except Exception:
    pass
layout["icon_11"] = [247, 2, 302, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_06_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-8/12_icon_Los_Angeles.png
try:
    _c12 = get_crop(12, 1440, 132)
    canvas.paste(_c12, (0, 1020), _c12)
except Exception:
    pass
layout["Los_Angeles"] = [0, 1020, 1440, 1152]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_06_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-8/13_icon_Miami.png
try:
    _c13 = get_crop(13, 1440, 132)
    canvas.paste(_c13, (0, 1200), _c13)
except Exception:
    pass
layout["Miami"] = [0, 1200, 1440, 1332]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_06_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-8/14_icon_United_Kingdom.png
try:
    _c14 = get_crop(14, 1440, 132)
    canvas.paste(_c14, (0, 2100), _c14)
except Exception:
    pass
layout["United_Kingdom"] = [0, 2100, 1440, 2232]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_06_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-8/15_icon_District_of_Columbia.png
try:
    _c15 = get_crop(15, 1440, 132)
    canvas.paste(_c15, (0, 1560), _c15)
except Exception:
    pass
layout["District_of_Columbia"] = [0, 1560, 1440, 1692]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_06_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-8/16_icon_Philadelphia.png
try:
    _c16 = get_crop(16, 1440, 132)
    canvas.paste(_c16, (0, 1920), _c16)
except Exception:
    pass
layout["Philadelphia"] = [0, 1920, 1440, 2052]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_06_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-8/17_icon_icon_17.png
try:
    _c17 = get_crop(17, 50, 63)
    canvas.paste(_c17, (382, 0), _c17)
except Exception:
    pass
layout["icon_17"] = [382, 0, 432, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_06_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-8/18_icon_District_of_Columbia.png
try:
    _c18 = get_crop(18, 1440, 132)
    canvas.paste(_c18, (0, 1560), _c18)
except Exception:
    pass
layout["District_of_Columbia"] = [0, 1560, 1440, 1692]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_06_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-8/19_text_9.41.png
try:
    _c19 = get_crop(19, 93, 50)
    canvas.paste(_c19, (18, 12), _c19)
except Exception:
    pass
layout["9.41"] = [18, 12, 111, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_06_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-8/20_text_Los_Angeles.png
try:
    _c20 = get_crop(20, 1344, 129)
    canvas.paste(_c20, (48, 264), _c20)
except Exception:
    pass
layout["Los_Angeles"] = [48, 264, 1392, 393]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_06_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-8/21_text_Nearby.png
try:
    _c21 = get_crop(21, 415, 114)
    canvas.paste(_c21, (48, 465), _c21)
except Exception:
    pass
layout["Nearby"] = [48, 465, 463, 579]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_06_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-8/22_text_Online_events.png
try:
    _c22 = get_crop(22, 452, 114)
    canvas.paste(_c22, (511, 465), _c22)
except Exception:
    pass
layout["Online_events"] = [511, 465, 963, 579]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_06_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-8/23_text_Current_location.png
try:
    _c23 = get_crop(23, 415, 114)
    canvas.paste(_c23, (48, 465), _c23)
except Exception:
    pass
layout["Current_location"] = [48, 465, 463, 579]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_06_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-8/24_text_Virtual_attendance.png
try:
    _c24 = get_crop(24, 452, 114)
    canvas.paste(_c24, (511, 465), _c24)
except Exception:
    pass
layout["Virtual_attendance"] = [511, 465, 963, 579]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_06_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-8/25_text_Found_locations.png
try:
    _c25 = get_crop(25, 311, 50)
    canvas.paste(_c25, (44, 740), _c25)
except Exception:
    pass
layout["Found_locations"] = [44, 740, 355, 790]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_06_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-8/26_text_New_York.png
try:
    _c26 = get_crop(26, 212, 55)
    canvas.paste(_c26, (44, 2288), _c26)
except Exception:
    pass
layout["New_York"] = [44, 2288, 256, 2343]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_06_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-8/27_text_New_York.png
try:
    _c27 = get_crop(27, 154, 38)
    canvas.paste(_c27, (47, 2353), _c27)
except Exception:
    pass
layout["New_York"] = [47, 2353, 201, 2391]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_06_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-8/28_text_Atlanta.png
try:
    _c28 = get_crop(28, 163, 52)
    canvas.paste(_c28, (44, 2468), _c28)
except Exception:
    pass
layout["Atlanta"] = [44, 2468, 207, 2520]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_06_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-8/29_text_Georgia.png
try:
    _c29 = get_crop(29, 133, 43)
    canvas.paste(_c29, (45, 2533), _c29)
except Exception:
    pass
layout["Georgia"] = [45, 2533, 178, 2576]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_06_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-8/30_clickable_New_York.png
try:
    _c30 = get_crop(30, 1440, 132)
    canvas.paste(_c30, (0, 2280), _c30)
except Exception:
    pass
layout["New_York"] = [0, 2280, 1440, 2412]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_06_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-8/31_clickable_Atlanta.png
try:
    _c31 = get_crop(31, 1440, 132)
    canvas.paste(_c31, (0, 2460), _c31)
except Exception:
    pass
layout["Atlanta"] = [0, 2460, 1440, 2592]
