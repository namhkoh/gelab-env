# page_id: page_eventbrite_f01eaa41f6284da09deb7ced3e4eea4e_09
# screenshot: 2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-11.png
# step_index: 9/11
# task: Open Eventbrite. Check out 'Sports' events. Apply filters for events happening this week. Select the first event. Check similar events and add the first similar event to favorite.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Background and structural UI elements for the Eventbrite-like page
w, h = canvas.size

# Colors
bg_color = (255, 255, 255)              # overall white background
status_bar_color = (200, 200, 200)      # light gray status bar
status_icon_bg = (230, 230, 230)        # subtle area under status
divider_light = (243, 244, 246)         # very light divider
divider = (225, 226, 230)               # standard divider
header_shadow = (220, 220, 224)         # shadow below header
ticket_card_fill = (255, 255, 255)      # ticket card background (white)
ticket_card_border = (52, 88, 255)      # blue border for ticket card
ticket_card_shadow = (235, 236, 240)    # shadow under ticket card
reserve_btn_color = (199, 67, 22)       # orange/red Reserve button
section_bg = (250, 250, 252)            # very light section background

# Clear / fill base background
draw.rectangle([(0, 0), (w, h)], fill=bg_color)

# Status bar (approx 56px high)
status_h = 56
draw.rectangle([(0, 0), (w, status_h)], fill=status_bar_color)
# subtle line under status
draw.line([(0, status_h), (w, status_h)], fill=divider, width=1)

# Header / toolbar area (from status_h to ~140px)
header_top = status_h
header_bottom = 140
draw.rectangle([(0, header_top), (w, header_bottom)], fill=bg_color)
# slight shadow/divider under header
draw.line([(24, header_bottom), (w - 24, header_bottom)], fill=header_shadow, width=1)

# Thin light divider under title area (separates top meta from content)
# place around y=240 to match screenshot spacing (above organizer section)
meta_div_y = 240
draw.line([(40, meta_div_y), (w - 40, meta_div_y)], fill=divider_light, width=2)

# Section separator before tags / related events area (subtle)
related_separator_y = 2000
draw.line([(40, related_separator_y), (w - 40, related_separator_y)], fill=divider_light, width=2)

# Ticket selection card area (rounded rectangle)
# approximate bounds based on detected icon positions
card_left = 48
card_right = w - 48
card_top = 2340
card_bottom = 2650
card_radius = 18

# simple shadow behind card
shadow_offset = 8
draw.rounded_rectangle(
    [(card_left + shadow_offset, card_top + shadow_offset),
     (card_right + shadow_offset, card_bottom + shadow_offset)],
    radius=card_radius, fill=ticket_card_shadow
)
# card background and blue border
draw.rounded_rectangle(
    [(card_left, card_top), (card_right, card_bottom)],
    radius=card_radius, fill=ticket_card_fill, outline=ticket_card_border, width=6
)

# Small inner divider line inside the card to hint content separation
inner_div_y = card_top + 84
draw.line([(card_left + 30, inner_div_y), (card_right - 30, inner_div_y)], fill=divider_light, width=1)

# Reserve button background band (full width, rounded)
reserve_left = 72
reserve_top = 2756
reserve_w = 1296
reserve_h = 132
reserve_radius = 10
reserve_rect = [ (reserve_left, reserve_top), (reserve_left + reserve_w, reserve_top + reserve_h) ]
draw.rounded_rectangle(reserve_rect, radius=reserve_radius, fill=reserve_btn_color)

# Subtle horizontal divider above ticket/card area
draw.line([(24, card_top - 40), (w - 24, card_top - 40)], fill=divider_light, width=1)

# Large subtle section background behind organizer area (to group content visually)
organizer_section_top = 240
organizer_section_bottom = 1180
draw.rectangle(
    [(24, organizer_section_top), (w - 24, organizer_section_bottom)],
    fill=section_bg
)
# overlay a very faint centered horizontal divider near the middle of organizer area
draw.line(
    [(60, organizer_section_top + 640), (w - 60, organizer_section_top + 640)],
    fill=divider_light, width=1
)

# Topmost thin full-width divider (visual upper boundary under status/header)
draw.line([(0, header_bottom + 6), (w, header_bottom + 6)], fill=divider, width=1)

# Subtle left/right padding vertical guides (very faint) to suggest layout gutters
gutters_color = (250, 250, 252)
draw.line([(40, header_bottom + 10), (40, h - 10)], fill=gutters_color, width=1)
draw.line([(w - 40, header_bottom + 10), (w - 40, h - 10)], fill=gutters_color, width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_09_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-11/00_icon_Follow.png
try:
    _c0 = get_crop(0, 384, 144)
    canvas.paste(_c0, (528, 1565), _c0)
except Exception:
    pass
layout["Follow"] = [528, 1565, 912, 1709]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_09_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-11/01_icon_sports.png
try:
    _c1 = get_crop(1, 190, 144)
    canvas.paste(_c1, (48, 2082), _c1)
except Exception:
    pass
layout["sports"] = [48, 2082, 238, 2226]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_09_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-11/02_icon_basement.png
try:
    _c2 = get_crop(2, 259, 144)
    canvas.paste(_c2, (641, 2082), _c2)
except Exception:
    pass
layout["basement"] = [641, 2082, 900, 2226]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_09_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-11/03_icon_backpacking.png
try:
    _c3 = get_crop(3, 307, 144)
    canvas.paste(_c3, (286, 2082), _c3)
except Exception:
    pass
layout["backpacking"] = [286, 2082, 593, 2226]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_09_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-11/04_icon_Reserve_a_spot.png
try:
    _c4 = get_crop(4, 1296, 132)
    canvas.paste(_c4, (72, 2756), _c4)
except Exception:
    pass
layout["Reserve_a_spot"] = [72, 2756, 1368, 2888]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_09_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-11/05_icon_clinic.png
try:
    _c5 = get_crop(5, 169, 144)
    canvas.paste(_c5, (948, 2082), _c5)
except Exception:
    pass
layout["clinic"] = [948, 2082, 1117, 2226]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_09_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-11/06_icon_More.png
try:
    _c6 = get_crop(6, 144, 144)
    canvas.paste(_c6, (1116, 108), _c6)
except Exception:
    pass
layout["More"] = [1116, 108, 1260, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_09_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-11/07_icon_Share.png
try:
    _c7 = get_crop(7, 144, 144)
    canvas.paste(_c7, (1260, 108), _c7)
except Exception:
    pass
layout["Share"] = [1260, 108, 1404, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_09_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-11/08_icon_4.36.png
try:
    _c8 = get_crop(8, 144, 144)
    canvas.paste(_c8, (36, 108), _c8)
except Exception:
    pass
layout["4.36"] = [36, 108, 180, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_09_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-11/09_icon_Decrease.png
try:
    _c9 = get_crop(9, 99, 96)
    canvas.paste(_c9, (996, 2444), _c9)
except Exception:
    pass
layout["Decrease"] = [996, 2444, 1095, 2540]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_09_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-11/10_icon_Increase.png
try:
    _c10 = get_crop(10, 96, 96)
    canvas.paste(_c10, (1224, 2444), _c10)
except Exception:
    pass
layout["Increase"] = [1224, 2444, 1320, 2540]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_09_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-11/11_icon_icon_11.png
try:
    _c11 = get_crop(11, 90, 102)
    canvas.paste(_c11, (1109, 2442), _c11)
except Exception:
    pass
layout["icon_11"] = [1109, 2442, 1199, 2544]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_09_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-11/12_icon_Backpacking_Clinic_W.png
try:
    _c12 = get_crop(12, 144, 144)
    canvas.paste(_c12, (36, 108), _c12)
except Exception:
    pass
layout["Backpacking_Clinic_W__"] = [36, 108, 180, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_09_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-11/13_icon_Iseme.png
try:
    _c13 = get_crop(13, 240, 240)
    canvas.paste(_c13, (600, 579), _c13)
except Exception:
    pass
layout["Iseme"] = [600, 579, 840, 819]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_09_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-11/14_icon_4.36.png
try:
    _c14 = get_crop(14, 63, 64)
    canvas.paste(_c14, (180, 1), _c14)
except Exception:
    pass
layout["4.36"] = [180, 1, 243, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_09_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-11/15_icon_icon_15.png
try:
    _c15 = get_crop(15, 60, 63)
    canvas.paste(_c15, (310, 2), _c15)
except Exception:
    pass
layout["icon_15"] = [310, 2, 370, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_09_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-11/16_icon_4.36.png
try:
    _c16 = get_crop(16, 63, 66)
    canvas.paste(_c16, (113, 0), _c16)
except Exception:
    pass
layout["4.36"] = [113, 0, 176, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_09_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-11/17_icon_icon_17.png
try:
    _c17 = get_crop(17, 52, 61)
    canvas.paste(_c17, (248, 3), _c17)
except Exception:
    pass
layout["icon_17"] = [248, 3, 300, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_09_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-11/18_icon_icon_18.png
try:
    _c18 = get_crop(18, 96, 62)
    canvas.paste(_c18, (1214, 1), _c18)
except Exception:
    pass
layout["icon_18"] = [1214, 1, 1310, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_09_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-11/19_icon_icon_19.png
try:
    _c19 = get_crop(19, 56, 64)
    canvas.paste(_c19, (1317, 0), _c19)
except Exception:
    pass
layout["icon_19"] = [1317, 0, 1373, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_09_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-11/20_icon_icon_20.png
try:
    _c20 = get_crop(20, 48, 64)
    canvas.paste(_c20, (383, 2), _c20)
except Exception:
    pass
layout["icon_20"] = [383, 2, 431, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_09_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-11/21_icon_Free.png
try:
    _c21 = get_crop(21, 139, 127)
    canvas.paste(_c21, (97, 2565), _c21)
except Exception:
    pass
layout["Free"] = [97, 2565, 236, 2692]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_09_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-11/22_icon_Free.png
try:
    _c22 = get_crop(22, 75, 72)
    canvas.paste(_c22, (249, 2588), _c22)
except Exception:
    pass
layout["Free"] = [249, 2588, 324, 2660]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_09_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-11/23_text_4.36.png
try:
    _c23 = get_crop(23, 89, 45)
    canvas.paste(_c23, (22, 15), _c23)
except Exception:
    pass
layout["4.36"] = [22, 15, 111, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_09_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-11/24_text_94703.png
try:
    _c24 = get_crop(24, 156, 54)
    canvas.paste(_c24, (137, 278), _c24)
except Exception:
    pass
layout["94703"] = [137, 278, 293, 332]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_09_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-11/25_text_Organized_by.png
try:
    _c25 = get_crop(25, 468, 144)
    canvas.paste(_c25, (486, 938), _c25)
except Exception:
    pass
layout["Organized_by"] = [486, 938, 954, 1082]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_09_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-11/26_text_Sports_Basement.png
try:
    _c26 = get_crop(26, 468, 144)
    canvas.paste(_c26, (486, 938), _c26)
except Exception:
    pass
layout["Sports_Basement"] = [486, 938, 954, 1082]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_09_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-11/27_text_2.2k.png
try:
    _c27 = get_crop(27, 132, 56)
    canvas.paste(_c27, (655, 1136), _c27)
except Exception:
    pass
layout["2.2k"] = [655, 1136, 787, 1192]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_09_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-11/28_text_Followers.png
try:
    _c28 = get_crop(28, 186, 51)
    canvas.paste(_c28, (628, 1207), _c28)
except Exception:
    pass
layout["Followers"] = [628, 1207, 814, 1258]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_09_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-11/29_text_Sports_Basement_is_a_sporting.png
try:
    _c29 = get_crop(29, 384, 144)
    canvas.paste(_c29, (528, 1565), _c29)
except Exception:
    pass
layout["Sports_Basement_is_a_spor"] = [528, 1565, 912, 1709]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_09_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-11/30_text_retailer_with_12.png
try:
    _c30 = get_crop(30, 290, 50)
    canvas.paste(_c30, (914, 1332), _c30)
except Exception:
    pass
layout["retailer_with_12"] = [914, 1332, 1204, 1382]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_09_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-11/31_text_brands_at_basement_prices._We_are_a_comm.png
try:
    _c31 = get_crop(31, 384, 144)
    canvas.paste(_c31, (528, 1565), _c31)
except Exception:
    pass
layout["brands_at_basement_prices"] = [528, 1565, 912, 1709]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_09_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-11/32_text_Related_to_this_event.png
try:
    _c32 = get_crop(32, 307, 144)
    canvas.paste(_c32, (286, 2082), _c32)
except Exception:
    pass
layout["Related_to_this_event"] = [286, 2082, 593, 2226]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_09_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-11/33_text_General_Admission.png
try:
    _c33 = get_crop(33, 75, 72)
    canvas.paste(_c33, (249, 2588), _c33)
except Exception:
    pass
layout["General_Admission"] = [249, 2588, 324, 2660]
