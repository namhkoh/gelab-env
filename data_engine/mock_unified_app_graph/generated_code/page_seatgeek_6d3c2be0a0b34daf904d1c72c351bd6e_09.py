# page_id: page_seatgeek_6d3c2be0a0b34daf904d1c72c351bd6e_09
# screenshot: 2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-12.png
# step_index: 9/9
# task: Open SeatGeek. Look up "Phoenix Suns" tickets for next upcoming event. Which section are tickets in?
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Canvas and draw are provided (PIL Image and ImageDraw). Fonts provided: font_sm, font_md, font_lg, font_xl
w, h = canvas.size

# --- Background fill (dominant canvas color is white/light) ---
draw.rectangle([(0, 0), (w, h)], fill="#ffffff")

# --- Top image/content area background (dark area for the stadium photo) ---
img_area_h = 1084  # matches detected clickable/image region height
draw.rectangle([(0, 0), (w, img_area_h)], fill="#0b2b44")  # deep bluish placeholder for photo area

# --- Status bar (approx ~50-84px high) ---
status_h = 84
draw.rectangle([(0, 0), (w, status_h)], fill="#071428")  # dark overlay for status icons area

# subtle thin divider at bottom of status bar
draw.line([(0, status_h), (w, status_h)], fill="#071428", width=1)

# --- White content card that overlaps the image (rounded top corners) ---
card_margin = 24
card_top = 900  # where white content starts (overlaps image)
card_bottom = h - 120
card_box = [ (card_margin, card_top), (w - card_margin, card_bottom) ]
draw.rounded_rectangle(card_box, radius=32, fill="#ffffff", outline=None)

# subtle top hairline shadow to separate from image
draw.line([(card_margin+1, card_top), (w-card_margin-1, card_top)], fill="#e6e6e6", width=2)

# slight outer shadow line just below top edge
draw.line([(card_margin+1, card_top+4), (w-card_margin-1, card_top+4)], fill="#f2f2f2", width=1)

# --- Primary horizontal dividers within the card ---
divider_color = "#e9e9e9"
# Divider under price/fees block
draw.line([(card_margin+24, card_top+320), (w-card_margin-24, card_top+320)], fill=divider_color, width=1)
# Divider under event details block
draw.line([(card_margin+24, card_top+560), (w-card_margin-24, card_top+560)], fill=divider_color, width=1)

# Separators for list items (2 e-tickets, add payment, add contact)
list_start_y = card_top + 760
item_height = 120
for i in range(4):
    y = list_start_y + i * item_height
    draw.line([(card_margin+24, y), (w-card_margin-24, y)], fill=divider_color, width=1)

# --- Light gray background strip for the final guarantee section ---
guarantee_top = card_top + 1400
guarantee_bottom = guarantee_top + 360
guarantee_box = [(card_margin+12, guarantee_top), (w-card_margin-12, guarantee_bottom)]
draw.rounded_rectangle(guarantee_box, radius=16, fill="#fbfbfb", outline=None)

# subtle divider lines inside guarantee box
draw.line([(card_margin+40, guarantee_top+110), (w-card_margin-40, guarantee_top+110)], fill=divider_color, width=1)

# small top and bottom separators around the guarantee area
draw.line([(card_margin+12, guarantee_top-12), (w-card_margin-12, guarantee_top-12)], fill=divider_color, width=1)
draw.line([(card_margin+12, guarantee_bottom+6), (w-card_margin-12, guarantee_bottom+6)], fill=divider_color, width=1)

# --- Tiny visual accents (pill/badge background behind 'Amazing Deal' area) ---
# We draw only the colored pill background (no text) to indicate structure.
pill_x = card_margin + 108
pill_y = card_top + 320
pill_w = 140
pill_h = 48
draw.rounded_rectangle([(pill_x, pill_y), (pill_x + pill_w, pill_y + pill_h)], radius=12, fill="#e9f7ec")

# --- Subtle horizontal rules further down to break sections ---
draw.line([(card_margin+24, card_top+980), (w-card_margin-24, card_top+980)], fill=divider_color, width=1)
draw.line([(card_margin+24, card_top+1160), (w-card_margin-24, card_top+1160)], fill=divider_color, width=1)

# --- Bottom area filler (footer safe area) ---
footer_top = h - 120
draw.rectangle([(0, footer_top), (w, h)], fill="#ffffff")
# slight top divider for footer
draw.line([(card_margin, footer_top), (w-card_margin, footer_top)], fill="#f0f0f0", width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_09_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-12/00_icon_Target_Center_Minneapolis_MN.png
try:
    _c0 = get_crop(0, 1260, 193)
    canvas.paste(_c0, (90, 2029), _c0)
except Exception:
    pass
layout["Target_Center;_Minneapoli"] = [90, 2029, 1350, 2222]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_09_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-12/01_icon_Every_ticket_is_protected_If_your_event_.png
try:
    _c1 = get_crop(1, 1260, 288)
    canvas.paste(_c1, (90, 2610), _c1)
except Exception:
    pass
layout["Every_ticket_is_protected"] = [90, 2610, 1350, 2898]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_09_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-12/02_icon_icon_2.png
try:
    _c2 = get_crop(2, 67, 66)
    canvas.paste(_c2, (1227, 2663), _c2)
except Exception:
    pass
layout["icon_2"] = [1227, 2663, 1294, 2729]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_09_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-12/03_icon_icon_3.png
try:
    _c3 = get_crop(3, 67, 75)
    canvas.paste(_c3, (1147, 0), _c3)
except Exception:
    pass
layout["icon_3"] = [1147, 0, 1214, 75]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_09_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-12/04_icon_2_tickets.png
try:
    _c4 = get_crop(4, 96, 96)
    canvas.paste(_c4, (60, 928), _c4)
except Exception:
    pass
layout["2_tickets"] = [60, 928, 156, 1024]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_09_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-12/05_icon_7.07_Wy.png
try:
    _c5 = get_crop(5, 144, 144)
    canvas.paste(_c5, (18, 84), _c5)
except Exception:
    pass
layout["7.07_Wy"] = [18, 84, 162, 228]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_09_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-12/06_icon_Amazing_Deal.png
try:
    _c6 = get_crop(6, 422, 68)
    canvas.paste(_c6, (133, 1219), _c6)
except Exception:
    pass
layout["Amazing_Deal"] = [133, 1219, 555, 1287]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_09_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-12/07_icon_Add_payment_method.png
try:
    _c7 = get_crop(7, 1260, 169)
    canvas.paste(_c7, (90, 2222), _c7)
except Exception:
    pass
layout["Add_payment_method"] = [90, 2222, 1350, 2391]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_09_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-12/08_text_7.07_Wy.png
try:
    _c8 = get_crop(8, 144, 144)
    canvas.paste(_c8, (18, 84), _c8)
except Exception:
    pass
layout["7.07_Wy"] = [18, 84, 162, 228]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_09_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-12/09_text_Includes_S83_in_fees_per_ticket.png
try:
    _c9 = get_crop(9, 620, 61)
    canvas.paste(_c9, (129, 1440), _c9)
except Exception:
    pass
layout["Includes_S83_in_fees_per_"] = [129, 1440, 749, 1501]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_09_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-12/10_text_2_e-tickets.png
try:
    _c10 = get_crop(10, 253, 52)
    canvas.paste(_c10, (278, 2100), _c10)
except Exception:
    pass
layout["2_e-tickets"] = [278, 2100, 531, 2152]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_09_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-12/11_text_Add_contact_info.png
try:
    _c11 = get_crop(11, 1260, 169)
    canvas.paste(_c11, (90, 2391), _c11)
except Exception:
    pass
layout["Add_contact_info"] = [90, 2391, 1350, 2560]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_09_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-12/12_clickable_Likely_opens_a_search_function_to_find_o.png
try:
    _c12 = get_crop(12, 1440, 1084)
    canvas.paste(_c12, (0, 0), _c12)
except Exception:
    pass
layout["Likely_opens_a_search_fun"] = [0, 0, 1440, 1084]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_09_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-12/13_clickable_Share.png
try:
    _c13 = get_crop(13, 144, 144)
    canvas.paste(_c13, (1272, 84), _c13)
except Exception:
    pass
layout["Share"] = [1272, 84, 1416, 228]
