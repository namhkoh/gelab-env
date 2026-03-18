# page_id: page_eventbrite_f1e087441f9e44d997c2a58b9c8b0258_04
# screenshot: 2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-6.png
# step_index: 4/10
# task: Open Eventbrite. Find the 'Arts' category. Select events that are available for this weekend. From the results, open the first item and add it to favorite. Follow the organizer.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw UI background and structural elements for the filters page
# Uses provided: canvas (1440x2960 RGB), draw (ImageDraw), font_sm,font_md,font_lg,font_xl

w, h = canvas.size

# Colors
bg_white = (250, 250, 251)        # very slightly off-white background
status_bar_gray = (190, 190, 190) # top status bar
header_divider = (235, 236, 240)  # subtle divider under header
section_separator = (240, 241, 245) # subtle horizontal separators
muted_shadow = (230,230,235)

# Fill overall background (ensures consistent off-white)
draw.rectangle([(0,0),(w,h)], fill=bg_white)

# Status bar area (top system bar)
status_h = 72
draw.rectangle([(0,0),(w,status_h)], fill=status_bar_gray)

# Header area (toolbar) - below status bar
header_top = status_h
header_bottom = 160
draw.rectangle([(0,header_top),(w,header_bottom)], fill=bg_white)

# Soft shadow line under status bar (between status and header)
draw.line([(0, header_top),(w, header_top)], fill=muted_shadow, width=1)

# Divider under header
draw.line([(20, header_bottom),(w-20, header_bottom)], fill=header_divider, width=1)

# Section separators between primary filter groups
# Positions approximate based on UI crop:
separators_y = [560, 1000, 1420, 1700, 1888]
for y in separators_y:
    draw.line([(36, y), (w-36, y)], fill=section_separator, width=1)

# Subtle grouped card backgrounds for larger grouping areas (no content drawn)
# Category/Event type/Languages groups use only very light rounded rectangles as structure hints
# REMOVED: from PIL import ImageFilter, Image

def rounded_rect(coord, radius, fill):
    # Draw rounded rectangle by ImageDraw rounded_rectangle if available
    try:
        draw.rounded_rectangle(coord, radius=radius, fill=fill)
    except Exception:
        # Fallback: draw normal rectangle
        draw.rectangle(coord, fill=fill)

group_bg = (247, 249, 252)  # extremely light bluish card tint
# Categories group background (behind chips area)
rounded_rect((24, 300, w-24, 520), radius=18, fill=group_bg)
# Event type group background
rounded_rect((24, 746, w-24, 964), radius=18, fill=group_bg)
# Languages group background
rounded_rect((24, 1188, w-24, 1410), radius=18, fill=group_bg)

# Price / toggle area background hint
rounded_rect((24, 1560, w-24, 1696), radius=12, fill=(255,255,255))  # keep white but define boundary
draw.line([(36,1696),(w-36,1696)], fill=section_separator, width=1)

# Sort-by segmented control background (subtle outer container)
seg_top = 2024 - 24
seg_bottom = seg_top + 200
seg_left = 36
seg_right = w - 36
rounded_rect((seg_left, seg_top, seg_right, seg_bottom), radius=16, fill=(250,250,253))
# inner subtle shadow under segmented control
draw.line([(seg_left+6, seg_bottom),(seg_right-6, seg_bottom)], fill=header_divider, width=1)

# Bottom safe area / faint top border above the sticky apply bar (do not draw the apply bar itself)
apply_bar_top = 2768 - 12
draw.line([(24, apply_bar_top), (w-24, apply_bar_top)], fill=section_separator, width=2)
# add very faint rounded outline to indicate area reserved for the apply button (actual button will be pasted)
draw.rounded_rectangle([(24, apply_bar_top+8), (w-24, apply_bar_top+160)], radius=12, outline=(225,226,230), width=2, fill=None)

# Small decorative shadows to separate major blocks (soft, low-opacity simulated with lines)
shadow_lines = [ (header_bottom+8, 1), (560+8, 1), (1000+8, 1), (1420+8, 1) ]
for y, wth in shadow_lines:
    draw.line([(36, y), (w-36, y)], fill=(245,245,247), width=wth)

# Done drawing structural elements. (No text or icons drawn.)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_04_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-6/00_icon_Food_Drink.png
try:
    _c0 = get_crop(0, 312, 144)
    canvas.paste(_c0, (512, 383), _c0)
except Exception:
    pass
layout["Food_&_Drink"] = [512, 383, 824, 527]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_04_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-6/01_icon_Music.png
try:
    _c1 = get_crop(1, 187, 135)
    canvas.paste(_c1, (36, 383), _c1)
except Exception:
    pass
layout["Music"] = [36, 383, 223, 518]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_04_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-6/02_icon_Community.png
try:
    _c2 = get_crop(2, 294, 144)
    canvas.paste(_c2, (848, 383), _c2)
except Exception:
    pass
layout["Community"] = [848, 383, 1142, 527]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_04_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-6/03_icon_French.png
try:
    _c3 = get_crop(3, 205, 144)
    canvas.paste(_c3, (768, 1275), _c3)
except Exception:
    pass
layout["French"] = [768, 1275, 973, 1419]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_04_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-6/04_icon_Arts.png
try:
    _c4 = get_crop(4, 152, 144)
    canvas.paste(_c4, (1166, 383), _c4)
except Exception:
    pass
layout["Arts"] = [1166, 383, 1318, 527]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_04_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-6/05_icon_Spanish.png
try:
    _c5 = get_crop(5, 225, 144)
    canvas.paste(_c5, (519, 1275), _c5)
except Exception:
    pass
layout["Spanish"] = [519, 1275, 744, 1419]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_04_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-6/06_icon_Business.png
try:
    _c6 = get_crop(6, 241, 135)
    canvas.paste(_c6, (247, 383), _c6)
except Exception:
    pass
layout["Business"] = [247, 383, 488, 518]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_04_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-6/07_icon_Expo.png
try:
    _c7 = get_crop(7, 167, 144)
    canvas.paste(_c7, (614, 829), _c7)
except Exception:
    pass
layout["Expo"] = [614, 829, 781, 973]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_04_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-6/08_icon_Seminar.png
try:
    _c8 = get_crop(8, 232, 144)
    canvas.paste(_c8, (358, 829), _c8)
except Exception:
    pass
layout["Seminar"] = [358, 829, 590, 973]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_04_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-6/09_icon_Italian.png
try:
    _c9 = get_crop(9, 191, 144)
    canvas.paste(_c9, (997, 1275), _c9)
except Exception:
    pass
layout["Italian"] = [997, 1275, 1188, 1419]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_04_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-6/10_icon_Convention.png
try:
    _c10 = get_crop(10, 293, 144)
    canvas.paste(_c10, (805, 829), _c10)
except Exception:
    pass
layout["Convention"] = [805, 829, 1098, 973]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_04_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-6/11_icon_Festival.png
try:
    _c11 = get_crop(11, 219, 144)
    canvas.paste(_c11, (1122, 829), _c11)
except Exception:
    pass
layout["Festival"] = [1122, 829, 1341, 973]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_04_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-6/12_icon_German.png
try:
    _c12 = get_crop(12, 225, 135)
    canvas.paste(_c12, (270, 1275), _c12)
except Exception:
    pass
layout["German"] = [270, 1275, 495, 1410]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_04_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-6/13_icon_English.png
try:
    _c13 = get_crop(13, 210, 135)
    canvas.paste(_c13, (36, 1275), _c13)
except Exception:
    pass
layout["English"] = [36, 1275, 246, 1410]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_04_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-6/14_icon_Conference.png
try:
    _c14 = get_crop(14, 298, 135)
    canvas.paste(_c14, (36, 829), _c14)
except Exception:
    pass
layout["Conference"] = [36, 829, 334, 964]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_04_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-6/15_icon_Apply_filters_1.png
try:
    _c15 = get_crop(15, 1344, 144)
    canvas.paste(_c15, (48, 2768), _c15)
except Exception:
    pass
layout["Apply_filters_(1)"] = [48, 2768, 1392, 2912]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_04_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-6/16_icon_Date.png
try:
    _c16 = get_crop(16, 660, 144)
    canvas.paste(_c16, (726, 2024), _c16)
except Exception:
    pass
layout["Date"] = [726, 2024, 1386, 2168]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_04_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-6/17_icon_Relevance.png
try:
    _c17 = get_crop(17, 660, 144)
    canvas.paste(_c17, (54, 2024), _c17)
except Exception:
    pass
layout["Relevance"] = [54, 2024, 714, 2168]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_04_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-6/18_icon_4.32.png
try:
    _c18 = get_crop(18, 144, 144)
    canvas.paste(_c18, (12, 72), _c18)
except Exception:
    pass
layout["4.32"] = [12, 72, 156, 216]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_04_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-6/19_icon_4.32.png
try:
    _c19 = get_crop(19, 61, 64)
    canvas.paste(_c19, (179, 2), _c19)
except Exception:
    pass
layout["4.32"] = [179, 2, 240, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_04_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-6/20_icon_icon_20.png
try:
    _c20 = get_crop(20, 64, 62)
    canvas.paste(_c20, (308, 3), _c20)
except Exception:
    pass
layout["icon_20"] = [308, 3, 372, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_04_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-6/21_icon_4.32.png
try:
    _c21 = get_crop(21, 64, 65)
    canvas.paste(_c21, (112, 1), _c21)
except Exception:
    pass
layout["4.32"] = [112, 1, 176, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_04_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-6/22_icon_Clear_all.png
try:
    _c22 = get_crop(22, 99, 65)
    canvas.paste(_c22, (1211, 0), _c22)
except Exception:
    pass
layout["Clear_all"] = [1211, 0, 1310, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_04_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-6/23_icon_Clear_all.png
try:
    _c23 = get_crop(23, 56, 67)
    canvas.paste(_c23, (1318, 0), _c23)
except Exception:
    pass
layout["Clear_all"] = [1318, 0, 1374, 67]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_04_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-6/24_icon_icon_24.png
try:
    _c24 = get_crop(24, 51, 62)
    canvas.paste(_c24, (248, 2), _c24)
except Exception:
    pass
layout["icon_24"] = [248, 2, 299, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_04_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-6/25_icon_Toggle_to_filter_only_free_events.png
try:
    _c25 = get_crop(25, 144, 144)
    canvas.paste(_c25, (1248, 1729), _c25)
except Exception:
    pass
layout["Toggle_to_filter_only_fre"] = [1248, 1729, 1392, 1873]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_04_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-6/26_icon_Clear_all.png
try:
    _c26 = get_crop(26, 178, 144)
    canvas.paste(_c26, (1214, 72), _c26)
except Exception:
    pass
layout["Clear_all"] = [1214, 72, 1392, 216]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_04_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-6/27_text_4.32.png
try:
    _c27 = get_crop(27, 89, 45)
    canvas.paste(_c27, (22, 15), _c27)
except Exception:
    pass
layout["4.32"] = [22, 15, 111, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_04_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-6/28_text_Filters.png
try:
    _c28 = get_crop(28, 180, 66)
    canvas.paste(_c28, (631, 116), _c28)
except Exception:
    pass
layout["Filters"] = [631, 116, 811, 182]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_04_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-6/29_text_Categories.png
try:
    _c29 = get_crop(29, 187, 135)
    canvas.paste(_c29, (36, 383), _c29)
except Exception:
    pass
layout["Categories"] = [36, 383, 223, 518]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_04_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-6/30_text_Show_all_categories.png
try:
    _c30 = get_crop(30, 516, 144)
    canvas.paste(_c30, (0, 518), _c30)
except Exception:
    pass
layout["Show_all_categories"] = [0, 518, 516, 662]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_04_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-6/31_text_Event_type.png
try:
    _c31 = get_crop(31, 298, 135)
    canvas.paste(_c31, (36, 829), _c31)
except Exception:
    pass
layout["Event_type"] = [36, 829, 334, 964]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_04_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-6/32_text_Show_all_event_types.png
try:
    _c32 = get_crop(32, 535, 144)
    canvas.paste(_c32, (0, 964), _c32)
except Exception:
    pass
layout["Show_all_event_types"] = [0, 964, 535, 1108]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_04_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-6/33_text_Languages.png
try:
    _c33 = get_crop(33, 210, 135)
    canvas.paste(_c33, (36, 1275), _c33)
except Exception:
    pass
layout["Languages"] = [36, 1275, 246, 1410]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_04_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-6/34_text_Show_all_languages.png
try:
    _c34 = get_crop(34, 511, 144)
    canvas.paste(_c34, (0, 1410), _c34)
except Exception:
    pass
layout["Show_all_languages"] = [0, 1410, 511, 1554]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_04_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-6/35_text_Price.png
try:
    _c35 = get_crop(35, 149, 63)
    canvas.paste(_c35, (45, 1613), _c35)
except Exception:
    pass
layout["Price"] = [45, 1613, 194, 1676]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_04_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-6/36_text_Only_free_events.png
try:
    _c36 = get_crop(36, 660, 144)
    canvas.paste(_c36, (54, 2024), _c36)
except Exception:
    pass
layout["Only_free_events"] = [54, 2024, 714, 2168]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_04_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-6/37_text_Sort_by.png
try:
    _c37 = get_crop(37, 206, 75)
    canvas.paste(_c37, (42, 1931), _c37)
except Exception:
    pass
layout["Sort_by"] = [42, 1931, 248, 2006]
