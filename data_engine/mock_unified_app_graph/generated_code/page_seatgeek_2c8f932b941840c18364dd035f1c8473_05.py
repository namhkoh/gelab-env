# page_id: page_seatgeek_2c8f932b941840c18364dd035f1c8473_05
# screenshot: 2024_4_22_17_5_2c8f932b941840c18364dd035f1c8473-8.png
# step_index: 5/8
# task: Open SeatGeek. Search "Beatles Love". Select the soonest upcoming event. Choose 2 tickets and continue. What is the lowest price for each ticket?
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw UI background and structure for the mobile page (canvas and draw are provided)

w, h = canvas.size

# Base background (dominant color: white)
draw.rectangle([(0, 0), (w, h)], fill="#FFFFFF")

# Top image/banner area (dark, matches the screenshot hero image area)
banner_h = 930
draw.rectangle([(0, 0), (w, banner_h)], fill="#17141A")

# Status bar area (~50px). Keep a darker strip at the very top for status icons to be pasted on later.
status_h = 110  # a bit larger to match phone status area
draw.rectangle([(0, 0), (w, status_h)], fill="#0E0E0E")

# Subtle gradient-ish band under banner to suggest image fade (drawn as two thin rectangles)
draw.rectangle([(0, banner_h - 8), (w, banner_h)], fill="#0B0A0B")
draw.rectangle([(0, banner_h), (w, banner_h + 2)], fill="#EDEDED")

# Main white content card under the banner (rounded)
card_margin = 24
card_top = banner_h - 30
card_bottom = 1420
draw.rounded_rectangle(
    [(card_margin, card_top), (w - card_margin, card_bottom)],
    radius=16,
    fill="#FFFFFF",
    outline=None
)

# Small divider line under the title area (subtle)
divider_y = card_top + 120
draw.line([(card_margin + 12, divider_y), (w - card_margin - 12, divider_y)], fill="#E6E6E6", width=1)

# Section separator under the "Track" card / message area
section_sep_y = 1560
draw.line([(card_margin, section_sep_y), (w - card_margin, section_sep_y)], fill="#F0F0F0", width=1)

# All Shows list background area (large white surface)
list_top = 1660
list_bottom = h - 40
draw.rounded_rectangle(
    [(0, list_top), (w, list_bottom)],
    radius=0,
    fill="#FFFFFF",
    outline=None
)

# Shadow line above the list to separate from above content
draw.line([(12, list_top), (w - 12, list_top)], fill="#EAEAEA", width=2)

# Draw separators for each listed show row (use detected approximate row tops)
row_tops = [1785, 2078, 2371, 2664]
for y in row_tops:
    # separator line across content area leaving some left padding for date pill (icons will be pasted above)
    draw.line([(60, y - 10), (w - 60, y - 10)], fill="#F0F0F0", width=1)

# Add subtle rounded card backgrounds behind each section header area (no text)
# Header under image (thin subtle card)
hdr_card = (card_margin + 6, banner_h + 6, w - card_margin - 6, banner_h + 110)
draw.rounded_rectangle(hdr_card, radius=12, fill="#FFFFFF")

# Very light divider under header card
draw.line([(card_margin + 20, hdr_card[3] + 6), (w - card_margin - 20, hdr_card[3] + 6)], fill="#F3F3F3", width=1)

# Bottom safe area subtle fill
safe_h = 40
draw.rectangle([(0, h - safe_h), (w, h)], fill="#FFFFFF")

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c8f932b941840c18364dd035f1c8473/step_05_2024_4_22_17_5_2c8f932b941840c18364dd035f1c8473-8/00_icon_Track_Now.png
try:
    _c0 = get_crop(0, 337, 153)
    canvas.paste(_c0, (60, 1376), _c0)
except Exception:
    pass
layout["Track_Now"] = [60, 1376, 397, 1529]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c8f932b941840c18364dd035f1c8473/step_05_2024_4_22_17_5_2c8f932b941840c18364dd035f1c8473-8/01_icon_Las_Vegas_NV.png
try:
    _c1 = get_crop(1, 1440, 293)
    canvas.paste(_c1, (0, 2078), _c1)
except Exception:
    pass
layout["Las_Vegas,_NV"] = [0, 2078, 1440, 2371]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c8f932b941840c18364dd035f1c8473/step_05_2024_4_22_17_5_2c8f932b941840c18364dd035f1c8473-8/02_icon_Vegas_NV.png
try:
    _c2 = get_crop(2, 1440, 293)
    canvas.paste(_c2, (0, 2371), _c2)
except Exception:
    pass
layout["Vegas,_NV"] = [0, 2371, 1440, 2664]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c8f932b941840c18364dd035f1c8473/step_05_2024_4_22_17_5_2c8f932b941840c18364dd035f1c8473-8/03_icon_24.png
try:
    _c3 = get_crop(3, 1440, 293)
    canvas.paste(_c3, (0, 2664), _c3)
except Exception:
    pass
layout["24"] = [0, 2664, 1440, 2957]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c8f932b941840c18364dd035f1c8473/step_05_2024_4_22_17_5_2c8f932b941840c18364dd035f1c8473-8/04_icon_Las_Vegas_NV.png
try:
    _c4 = get_crop(4, 1440, 293)
    canvas.paste(_c4, (0, 1785), _c4)
except Exception:
    pass
layout["Las_Vegas,_NV"] = [0, 1785, 1440, 2078]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c8f932b941840c18364dd035f1c8473/step_05_2024_4_22_17_5_2c8f932b941840c18364dd035f1c8473-8/05_icon_Track_this_performer.png
try:
    _c5 = get_crop(5, 144, 144)
    canvas.paste(_c5, (1104, 84), _c5)
except Exception:
    pass
layout["Track_this_performer"] = [1104, 84, 1248, 228]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c8f932b941840c18364dd035f1c8473/step_05_2024_4_22_17_5_2c8f932b941840c18364dd035f1c8473-8/06_icon_5.07_Wy.png
try:
    _c6 = get_crop(6, 144, 144)
    canvas.paste(_c6, (36, 84), _c6)
except Exception:
    pass
layout["5.07_Wy"] = [36, 84, 180, 228]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c8f932b941840c18364dd035f1c8473/step_05_2024_4_22_17_5_2c8f932b941840c18364dd035f1c8473-8/07_icon_Share_this_performer.png
try:
    _c7 = get_crop(7, 144, 144)
    canvas.paste(_c7, (1260, 84), _c7)
except Exception:
    pass
layout["Share_this_performer"] = [1260, 84, 1404, 228]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c8f932b941840c18364dd035f1c8473/step_05_2024_4_22_17_5_2c8f932b941840c18364dd035f1c8473-8/08_icon_24.png
try:
    _c8 = get_crop(8, 1440, 293)
    canvas.paste(_c8, (0, 2371), _c8)
except Exception:
    pass
layout["24"] = [0, 2371, 1440, 2664]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c8f932b941840c18364dd035f1c8473/step_05_2024_4_22_17_5_2c8f932b941840c18364dd035f1c8473-8/09_icon_23.png
try:
    _c9 = get_crop(9, 1440, 293)
    canvas.paste(_c9, (0, 2078), _c9)
except Exception:
    pass
layout["23"] = [0, 2078, 1440, 2371]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c8f932b941840c18364dd035f1c8473/step_05_2024_4_22_17_5_2c8f932b941840c18364dd035f1c8473-8/10_icon_23.png
try:
    _c10 = get_crop(10, 1440, 293)
    canvas.paste(_c10, (0, 1785), _c10)
except Exception:
    pass
layout["23"] = [0, 1785, 1440, 2078]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c8f932b941840c18364dd035f1c8473/step_05_2024_4_22_17_5_2c8f932b941840c18364dd035f1c8473-8/11_icon_Las_Vegas_NV.png
try:
    _c11 = get_crop(11, 1440, 293)
    canvas.paste(_c11, (0, 2664), _c11)
except Exception:
    pass
layout["Las_Vegas,_NV"] = [0, 2664, 1440, 2957]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c8f932b941840c18364dd035f1c8473/step_05_2024_4_22_17_5_2c8f932b941840c18364dd035f1c8473-8/12_icon_icon_12.png
try:
    _c12 = get_crop(12, 58, 71)
    canvas.paste(_c12, (1148, 2), _c12)
except Exception:
    pass
layout["icon_12"] = [1148, 2, 1206, 73]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c8f932b941840c18364dd035f1c8473/step_05_2024_4_22_17_5_2c8f932b941840c18364dd035f1c8473-8/13_icon_5.07_Wy.png
try:
    _c13 = get_crop(13, 170, 79)
    canvas.paste(_c13, (224, 0), _c13)
except Exception:
    pass
layout["5.07_Wy"] = [224, 0, 394, 79]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c8f932b941840c18364dd035f1c8473/step_05_2024_4_22_17_5_2c8f932b941840c18364dd035f1c8473-8/14_icon_icon_14.png
try:
    _c14 = get_crop(14, 57, 65)
    canvas.paste(_c14, (1315, 2), _c14)
except Exception:
    pass
layout["icon_14"] = [1315, 2, 1372, 67]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c8f932b941840c18364dd035f1c8473/step_05_2024_4_22_17_5_2c8f932b941840c18364dd035f1c8473-8/15_icon_icon_15.png
try:
    _c15 = get_crop(15, 99, 71)
    canvas.paste(_c15, (1212, 1), _c15)
except Exception:
    pass
layout["icon_15"] = [1212, 1, 1311, 72]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c8f932b941840c18364dd035f1c8473/step_05_2024_4_22_17_5_2c8f932b941840c18364dd035f1c8473-8/16_icon_Cirque_du_Soleil_The_Beatles.png
try:
    _c16 = get_crop(16, 1440, 126)
    canvas.paste(_c16, (0, 933), _c16)
except Exception:
    pass
layout["Cirque_du_Soleil:_The_Bea"] = [0, 933, 1440, 1059]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c8f932b941840c18364dd035f1c8473/step_05_2024_4_22_17_5_2c8f932b941840c18364dd035f1c8473-8/17_icon_icon_17.png
try:
    _c17 = get_crop(17, 80, 98)
    canvas.paste(_c17, (1308, 954), _c17)
except Exception:
    pass
layout["icon_17"] = [1308, 954, 1388, 1052]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c8f932b941840c18364dd035f1c8473/step_05_2024_4_22_17_5_2c8f932b941840c18364dd035f1c8473-8/18_icon_5.07_Wy.png
try:
    _c18 = get_crop(18, 56, 66)
    canvas.paste(_c18, (175, 0), _c18)
except Exception:
    pass
layout["5.07_Wy"] = [175, 0, 231, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c8f932b941840c18364dd035f1c8473/step_05_2024_4_22_17_5_2c8f932b941840c18364dd035f1c8473-8/19_icon_Las_Vegas_NV.png
try:
    _c19 = get_crop(19, 1440, 293)
    canvas.paste(_c19, (0, 1785), _c19)
except Exception:
    pass
layout["Las_Vegas,_NV"] = [0, 1785, 1440, 2078]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c8f932b941840c18364dd035f1c8473/step_05_2024_4_22_17_5_2c8f932b941840c18364dd035f1c8473-8/20_icon_5.07_Wy.png
try:
    _c20 = get_crop(20, 59, 65)
    canvas.paste(_c20, (112, 3), _c20)
except Exception:
    pass
layout["5.07_Wy"] = [112, 3, 171, 68]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c8f932b941840c18364dd035f1c8473/step_05_2024_4_22_17_5_2c8f932b941840c18364dd035f1c8473-8/21_text_No_Shows_near_New_York_NY.png
try:
    _c21 = get_crop(21, 337, 153)
    canvas.paste(_c21, (60, 1376), _c21)
except Exception:
    pass
layout["No_Shows_near_New_York,_N"] = [60, 1376, 397, 1529]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c8f932b941840c18364dd035f1c8473/step_05_2024_4_22_17_5_2c8f932b941840c18364dd035f1c8473-8/22_text_Track_Cirque_du_Soleil_The_Beatles.png
try:
    _c22 = get_crop(22, 337, 153)
    canvas.paste(_c22, (60, 1376), _c22)
except Exception:
    pass
layout["Track_Cirque_du_Soleil:_T"] = [60, 1376, 397, 1529]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c8f932b941840c18364dd035f1c8473/step_05_2024_4_22_17_5_2c8f932b941840c18364dd035f1c8473-8/23_text_Love_for_event_updates.png
try:
    _c23 = get_crop(23, 491, 56)
    canvas.paste(_c23, (826, 1286), _c23)
except Exception:
    pass
layout["Love_for_event_updates"] = [826, 1286, 1317, 1342]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c8f932b941840c18364dd035f1c8473/step_05_2024_4_22_17_5_2c8f932b941840c18364dd035f1c8473-8/24_text_AII_Shows.png
try:
    _c24 = get_crop(24, 249, 54)
    canvas.paste(_c24, (60, 1684), _c24)
except Exception:
    pass
layout["AII_Shows"] = [60, 1684, 309, 1738]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c8f932b941840c18364dd035f1c8473/step_05_2024_4_22_17_5_2c8f932b941840c18364dd035f1c8473-8/25_text_GROSSE.png
try:
    _c25 = get_crop(25, 157, 85)
    canvas.paste(_c25, (491, 0), _c25)
except Exception:
    pass
layout["GROSSE"] = [491, -1, 648, 84]
