# page_id: page_seatgeek_21b637d11bea46b8adb3c2efc9f03501_10
# screenshot: 2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-13.png
# step_index: 10/10
# task: Open SeatGeek and find the soonest upcoming NBA game in New York with "Nets", record the cheapest price in google keep notes.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw UI background and structural elements for a 1440x2960 mobile canvas.
# Available variables: canvas (PIL.Image), draw (PIL.ImageDraw), font_sm, font_md, font_lg, font_xl

w, h = canvas.size

# Colors
dark_bg = (18, 18, 18)         # dark area behind the hero image / photo
status_bar = (10, 10, 10)      # status bar band
card_shadow = (220, 220, 220)  # subtle shadow under cards
card_fill = (255, 255, 255)    # card / content background
divider = (230, 230, 230)      # thin separators
muted_divider = (245, 245, 245)

# Top photo/hero area (dark background to match the image area).
# The screenshot's large hero image area covers about the top 1084px (detected).
hero_h = 1084
draw.rectangle([0, 0, w, hero_h], fill=dark_bg)

# Status bar (approx ~50-84px high) - a slightly darker band at the very top.
status_h = 84
draw.rectangle([0, 0, w, status_h], fill=status_bar)

# Slight translucent-like top toolbar band (solid darker band to imply toolbar area)
toolbar_h = 160
draw.rectangle([0, 0, w, toolbar_h], fill=dark_bg)

# Price/details card that overlaps the bottom of the hero image.
# This is a rounded white card with a subtle shadow.
card_left = 40
card_right = w - 40
card_top = hero_h - 160   # card overlaps into the hero area
card_bottom = card_top + 360
card_radius = 28

# Shadow (offset)
shadow_offset = 8
draw.rounded_rectangle(
    [card_left + shadow_offset, card_top + shadow_offset, card_right + shadow_offset, card_bottom + shadow_offset],
    radius=card_radius,
    fill=card_shadow
)

# Main card
draw.rounded_rectangle([card_left, card_top, card_right, card_bottom], radius=card_radius, fill=card_fill)

# Inner subtle separators in the card (to delineate price/details regions)
sep_y1 = card_top + 140
sep_y2 = card_top + 220
draw.line([card_left + 24, sep_y1, card_right - 24, sep_y1], fill=divider, width=1)
draw.line([card_left + 24, sep_y2, card_right - 24, sep_y2], fill=divider, width=1)

# A very light divider under the card to smoothly connect to the list area
draw.line([card_left + 12, card_bottom, card_right - 12, card_bottom], fill=muted_divider, width=1)

# Main content/list area below the card (full-width white background)
content_top = card_bottom + 0
draw.rectangle([0, content_top, w, h], fill=card_fill)

# Grouped section card blocks for list items (visual grouping with subtle spacing)
# Large list region has multiple logical groups; draw subtle background distinction lines.
group_left = 60
group_right = w - 60

# Section separators (horizontal rules across the content area)
separators = [
    content_top + 140,  # after the first block (ticket details)
    content_top + 520,  # after the "notes" block
    content_top + 880,  # after the "2 e-tickets" block
    content_top + 1180, # another group boundary farther down
]
for y in separators:
    draw.line([group_left, y, group_right, y], fill=divider, width=1)

# Thin faint dividing margin near the very bottom area for the installments section
bottom_div_y = h - 340
draw.line([group_left, bottom_div_y, group_right, bottom_div_y], fill=muted_divider, width=1)

# Top-of-list small shadow under the card to imply elevation continuity
draw.rectangle([card_left + 12, card_bottom + 1, card_right - 12, card_bottom + 3], fill=card_shadow)

# A subtle left-aligned vertical guide (not visible UI element, but establishes structure spacing)
# (draw a very faint line for alignment of icons/content)
guide_x = group_left + 10
draw.line([guide_x, content_top + 40, guide_x, h - 200], fill=(245, 245, 245), width=1)

# End of structural drawing.
# Note: actual icons, texts, and interactive elements will be pasted on top of this structure.

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_10_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-13/00_icon_icon_0.png
try:
    _c0 = get_crop(0, 52, 67)
    canvas.paste(_c0, (1319, 3), _c0)
except Exception:
    pass
layout["icon_0"] = [1319, 3, 1371, 70]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_10_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-13/01_icon_icon_1.png
try:
    _c1 = get_crop(1, 68, 67)
    canvas.paste(_c1, (1213, 4), _c1)
except Exception:
    pass
layout["icon_1"] = [1213, 4, 1281, 71]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_10_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-13/02_icon_6.38.png
try:
    _c2 = get_crop(2, 144, 144)
    canvas.paste(_c2, (18, 84), _c2)
except Exception:
    pass
layout["6.38"] = [18, 84, 162, 228]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_10_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-13/03_icon_Madison_Square_Garden_New_York_NY.png
try:
    _c3 = get_crop(3, 1260, 252)
    canvas.paste(_c3, (90, 1571), _c3)
except Exception:
    pass
layout["Madison_Square_Garden,_Ne"] = [90, 1571, 1350, 1823]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_10_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-13/04_icon_icon_4.png
try:
    _c4 = get_crop(4, 60, 73)
    canvas.paste(_c4, (1149, 2), _c4)
except Exception:
    pass
layout["icon_4"] = [1149, 2, 1209, 75]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_10_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-13/05_icon_SEER.png
try:
    _c5 = get_crop(5, 55, 54)
    canvas.paste(_c5, (182, 9), _c5)
except Exception:
    pass
layout["SEER"] = [182, 9, 237, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_10_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-13/06_icon_Great_Deal.png
try:
    _c6 = get_crop(6, 353, 66)
    canvas.paste(_c6, (133, 1221), _c6)
except Exception:
    pass
layout["Great_Deal"] = [133, 1221, 486, 1287]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_10_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-13/07_icon_SEER.png
try:
    _c7 = get_crop(7, 57, 57)
    canvas.paste(_c7, (116, 8), _c7)
except Exception:
    pass
layout["SEER"] = [116, 8, 173, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_10_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-13/08_icon_icon_8.png
try:
    _c8 = get_crop(8, 47, 51)
    canvas.paste(_c8, (1261, 959), _c8)
except Exception:
    pass
layout["icon_8"] = [1261, 959, 1308, 1010]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_10_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-13/09_icon_Share.png
try:
    _c9 = get_crop(9, 144, 144)
    canvas.paste(_c9, (1272, 84), _c9)
except Exception:
    pass
layout["Share"] = [1272, 84, 1416, 228]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_10_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-13/10_icon_NEWYORKEMHS.png
try:
    _c10 = get_crop(10, 1440, 1084)
    canvas.paste(_c10, (0, 0), _c10)
except Exception:
    pass
layout["NEWYORKEMHS"] = [0, 0, 1440, 1084]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_10_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-13/11_icon_icon_11.png
try:
    _c11 = get_crop(11, 47, 50)
    canvas.paste(_c11, (1348, 959), _c11)
except Exception:
    pass
layout["icon_11"] = [1348, 959, 1395, 1009]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_10_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-13/12_icon_icon_12.png
try:
    _c12 = get_crop(12, 89, 85)
    canvas.paste(_c12, (128, 2464), _c12)
except Exception:
    pass
layout["icon_12"] = [128, 2464, 217, 2549]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_10_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-13/13_icon_3.png
try:
    _c13 = get_crop(13, 96, 96)
    canvas.paste(_c13, (60, 928), _c13)
except Exception:
    pass
layout["{3"] = [60, 928, 156, 1024]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_10_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-13/14_icon_Learn_more.png
try:
    _c14 = get_crop(14, 144, 138)
    canvas.paste(_c14, (1194, 2105), _c14)
except Exception:
    pass
layout["Learn_more"] = [1194, 2105, 1338, 2243]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_10_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-13/15_icon_2_tickets.png
try:
    _c15 = get_crop(15, 1260, 252)
    canvas.paste(_c15, (90, 1571), _c15)
except Exception:
    pass
layout["2_tickets"] = [90, 1571, 1350, 1823]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_10_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-13/16_text_6.38.png
try:
    _c16 = get_crop(16, 91, 45)
    canvas.paste(_c16, (20, 15), _c16)
except Exception:
    pass
layout["6.38"] = [20, 15, 111, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_10_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-13/17_text_Mobile_Transfer.png
try:
    _c17 = get_crop(17, 1260, 252)
    canvas.paste(_c17, (90, 1571), _c17)
except Exception:
    pass
layout["Mobile_Transfer"] = [90, 1571, 1350, 1823]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_10_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-13/18_text_Tickets_will_be_transferred_outside_of.png
try:
    _c18 = get_crop(18, 1260, 252)
    canvas.paste(_c18, (90, 1571), _c18)
except Exception:
    pass
layout["Tickets_will_be_transferr"] = [90, 1571, 1350, 1823]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_10_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-13/19_text_SeatGeek_to_your_contact_email.png
try:
    _c19 = get_crop(19, 1260, 252)
    canvas.paste(_c19, (90, 1571), _c19)
except Exception:
    pass
layout["SeatGeek_to_your_contact_"] = [90, 1571, 1350, 1823]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_10_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-13/20_text_Notes_from_the_seller.png
try:
    _c20 = get_crop(20, 491, 50)
    canvas.paste(_c20, (280, 2148), _c20)
except Exception:
    pass
layout["Notes_from_the_seller"] = [280, 2148, 771, 2198]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_10_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-13/21_text_WALLET_XFER.png
try:
    _c21 = get_crop(21, 302, 50)
    canvas.paste(_c21, (280, 2227), _c21)
except Exception:
    pass
layout["WALLET_XFER"] = [280, 2227, 582, 2277]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_10_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-13/22_text_2_e-tickets.png
try:
    _c22 = get_crop(22, 253, 52)
    canvas.paste(_c22, (278, 2481), _c22)
except Exception:
    pass
layout["2_e-tickets"] = [278, 2481, 531, 2533]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_10_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-13/23_text_Add_payment_method.png
try:
    _c23 = get_crop(23, 1260, 169)
    canvas.paste(_c23, (90, 2604), _c23)
except Exception:
    pass
layout["Add_payment_method"] = [90, 2604, 1350, 2773]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_10_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-13/24_text_Pay_in_installments.png
try:
    _c24 = get_crop(24, 1260, 187)
    canvas.paste(_c24, (90, 2773), _c24)
except Exception:
    pass
layout["Pay_in_installments"] = [90, 2773, 1350, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_10_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-13/25_text_4_interest-free_payments_or_as_low_as_S3.png
try:
    _c25 = get_crop(25, 1260, 187)
    canvas.paste(_c25, (90, 2773), _c25)
except Exception:
    pass
layout["4_interest-free_payments_"] = [90, 2773, 1350, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_10_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-13/26_clickable_2_e-tickets.png
try:
    _c26 = get_crop(26, 1260, 193)
    canvas.paste(_c26, (90, 2411), _c26)
except Exception:
    pass
layout["2_e-tickets"] = [90, 2411, 1350, 2604]
