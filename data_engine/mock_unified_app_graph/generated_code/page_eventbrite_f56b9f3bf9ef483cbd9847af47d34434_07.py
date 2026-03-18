# page_id: page_eventbrite_f56b9f3bf9ef483cbd9847af47d34434_07
# screenshot: 2024_4_24_17_3_f56b9f3bf9ef483cbd9847af47d34434-9.png
# step_index: 7/8
# task: Open Eventbrite. Look up "Gardening" events. Filter by events happening this week. Select the first event from the results. Follow the organizer and where is the event?
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw a status bar area (top)
draw.rectangle([(0, 0), (1440, 90)], fill=(210, 210, 210))  # light gray status bar

# Draw header/banner background (green blurred style)
# Simulate a vertical green gradient band across the top area where the hero image sits
header_top = 90
header_bottom = 520
for i in range(header_top, header_bottom):
    # interpolate between two greens
    t = (i - header_top) / max(1, (header_bottom - header_top - 1))
    # start green (deep) to lighter green
    r = int(35 + (170 - 35) * t)
    g = int(120 + (220 - 120) * t)
    b = int(60 + (150 - 60) * t)
    draw.line([(0, i), (1440, i)], fill=(r, g, b))

# Slight horizontal blur bands to mimic bokeh/blur on the edges
for offset, alpha in [(-140, 18), (-80, 12), (80, 12), (140, 18)]:
    band_top = header_top + 40 + offset//4
    band_bottom = header_top + 240 + offset//6
    color = (40, 140, 80)
    draw.rectangle([(0, band_top), (1440, band_bottom)], fill=color)

# Add a subtle white fade at the lower edge of the header to blend into content
fade_top = header_bottom - 30
fade_bottom = header_bottom + 40
for i in range(fade_top, fade_bottom):
    t = (i - fade_top) / max(1, (fade_bottom - fade_top - 1))
    # blend green -> white
    base_r, base_g, base_b = (120, 200, 120)
    r = int(base_r + (255 - base_r) * t)
    g = int(base_g + (255 - base_g) * t)
    b = int(base_b + (255 - base_b) * t)
    draw.line([(0, i), (1440, i)], fill=(r, g, b))

# Main content background (white)
draw.rectangle([(0, header_bottom), (1440, 2960)], fill=(255, 255, 255))

# Divider line under header
draw.line([(48, header_bottom + 8), (1392, header_bottom + 8)], fill=(230, 230, 235), width=2)

# Organizer / host card background (rounded rectangle)
org_card_top = 1000
org_card_bottom = 1160
org_card_rect = [48, org_card_top, 1392, org_card_bottom]
draw.rounded_rectangle(org_card_rect, radius=28, fill=(246, 245, 248), outline=None)

# subtle shadow under organizer card
shadow_top = org_card_bottom + 4
for i, a in enumerate(range(8, 2, -1)):
    y = shadow_top + i
    alpha = int(a * 3)
    # draw faint gray lines to emulate shadow
    draw.line([(50, y), (1390, y)], fill=(230, 230, 235))

# A thin section separator (used between details and about)
separator_y = 1500
draw.line([(48, separator_y), (1392, separator_y)], fill=(240, 240, 245), width=2)

# Another subtle divider further down
draw.line([(48, 1920), (1392, 1920)], fill=(244, 244, 246), width=1)

# Ticket selection card (rounded rectangle with blue border)
ticket_card_top = 2000
ticket_card_bottom = 2280  # keep above reserve button area
ticket_card_rect = [48, ticket_card_top, 1392, ticket_card_bottom]
draw.rounded_rectangle(ticket_card_rect, radius=20, fill=(255, 255, 255), outline=(60, 90, 255), width=6)

# Inner subtle background for ticket card (light gray)
inner_rect = [72, ticket_card_top + 28, 1368, ticket_card_bottom - 28]
draw.rounded_rectangle(inner_rect, radius=14, fill=(250, 250, 252), outline=None)

# Small control pill area on the right of the ticket card (just background shape)
control_rect = [1180, ticket_card_top + 40, 1368, ticket_card_bottom - 40]
draw.rounded_rectangle(control_rect, radius=14, fill=(245, 246, 255), outline=None)

# Subtle shadow under ticket card
for i in range(6):
    y = ticket_card_bottom + i
    shade = 240 + i
    draw.line([(52, y), (1388, y)], fill=(shade, shade, shade))

# Top-left content area accent: small pale lilac banner background (behind the "Ticket sales end soon" badge)
# NOTE: Only draw the background banner shape, not any badge text or icon.
badge_bg_rect = [36, 720, 320, 780]
draw.rounded_rectangle(badge_bg_rect, radius=20, fill=(245, 240, 255), outline=None)

# About section background hint (a faint rounded container behind the "About this event" area)
about_box = [36, 2060, 1392, 2180]
draw.rectangle(about_box, fill=(255, 255, 255))  # keep white but add top divider
draw.line([(48, 2060), (1392, 2060)], fill=(236, 236, 238), width=2)

# Bottom overall subtle top shadow to separate content from fixed bottom button area
bottom_button_top = 2324  # Reserve a spot area begins here (do not draw the button itself)
for i in range(8):
    y = bottom_button_top - 8 + i
    shade = 250 - i*2
    draw.line([(0, y), (1440, y)], fill=(shade, shade, shade))

# Final thin left/right content gutters (visual)
draw.line([(48, header_bottom + 60), (48, ticket_card_top - 60)], fill=(250, 250, 252), width=1)
draw.line([(1392, header_bottom + 60), (1392, ticket_card_top - 60)], fill=(250, 250, 252), width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f56b9f3bf9ef483cbd9847af47d34434/step_07_2024_4_24_17_3_f56b9f3bf9ef483cbd9847af47d34434-9/00_icon_Follow.png
try:
    _c0 = get_crop(0, 331, 144)
    canvas.paste(_c0, (1013, 1290), _c0)
except Exception:
    pass
layout["Follow"] = [1013, 1290, 1344, 1434]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f56b9f3bf9ef483cbd9847af47d34434/step_07_2024_4_24_17_3_f56b9f3bf9ef483cbd9847af47d34434-9/01_icon_More.png
try:
    _c1 = get_crop(1, 144, 144)
    canvas.paste(_c1, (1116, 108), _c1)
except Exception:
    pass
layout["More"] = [1116, 108, 1260, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f56b9f3bf9ef483cbd9847af47d34434/step_07_2024_4_24_17_3_f56b9f3bf9ef483cbd9847af47d34434-9/02_icon_Share.png
try:
    _c2 = get_crop(2, 144, 144)
    canvas.paste(_c2, (1260, 108), _c2)
except Exception:
    pass
layout["Share"] = [1260, 108, 1404, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f56b9f3bf9ef483cbd9847af47d34434/step_07_2024_4_24_17_3_f56b9f3bf9ef483cbd9847af47d34434-9/03_icon_5.10_my.png
try:
    _c3 = get_crop(3, 144, 144)
    canvas.paste(_c3, (36, 108), _c3)
except Exception:
    pass
layout["5.10_my"] = [36, 108, 180, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f56b9f3bf9ef483cbd9847af47d34434/step_07_2024_4_24_17_3_f56b9f3bf9ef483cbd9847af47d34434-9/04_icon_icon_4.png
try:
    _c4 = get_crop(4, 112, 106)
    canvas.paste(_c4, (988, 2439), _c4)
except Exception:
    pass
layout["icon_4"] = [988, 2439, 1100, 2545]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f56b9f3bf9ef483cbd9847af47d34434/step_07_2024_4_24_17_3_f56b9f3bf9ef483cbd9847af47d34434-9/05_icon_icon_5.png
try:
    _c5 = get_crop(5, 106, 103)
    canvas.paste(_c5, (1216, 2442), _c5)
except Exception:
    pass
layout["icon_5"] = [1216, 2442, 1322, 2545]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f56b9f3bf9ef483cbd9847af47d34434/step_07_2024_4_24_17_3_f56b9f3bf9ef483cbd9847af47d34434-9/06_icon_Reserve_a_spot.png
try:
    _c6 = get_crop(6, 1440, 636)
    canvas.paste(_c6, (0, 2324), _c6)
except Exception:
    pass
layout["Reserve_a_spot"] = [0, 2324, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f56b9f3bf9ef483cbd9847af47d34434/step_07_2024_4_24_17_3_f56b9f3bf9ef483cbd9847af47d34434-9/07_icon_icon_7.png
try:
    _c7 = get_crop(7, 90, 100)
    canvas.paste(_c7, (1109, 2444), _c7)
except Exception:
    pass
layout["icon_7"] = [1109, 2444, 1199, 2544]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f56b9f3bf9ef483cbd9847af47d34434/step_07_2024_4_24_17_3_f56b9f3bf9ef483cbd9847af47d34434-9/08_icon_Ticket_sales_end_soon.png
try:
    _c8 = get_crop(8, 547, 84)
    canvas.paste(_c8, (40, 753), _c8)
except Exception:
    pass
layout["Ticket_sales_end_soon"] = [40, 753, 587, 837]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f56b9f3bf9ef483cbd9847af47d34434/step_07_2024_4_24_17_3_f56b9f3bf9ef483cbd9847af47d34434-9/09_icon_The_organizer_will_review_refund_request.png
try:
    _c9 = get_crop(9, 1344, 144)
    canvas.paste(_c9, (48, 1517), _c9)
except Exception:
    pass
layout["The_organizer_will_review"] = [48, 1517, 1392, 1661]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f56b9f3bf9ef483cbd9847af47d34434/step_07_2024_4_24_17_3_f56b9f3bf9ef483cbd9847af47d34434-9/10_icon_430_Loudon_Rd_Concord_NH_USA.png
try:
    _c10 = get_crop(10, 293, 144)
    canvas.paste(_c10, (144, 1290), _c10)
except Exception:
    pass
layout["430_Loudon_Rd;_Concord;_N"] = [144, 1290, 437, 1434]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f56b9f3bf9ef483cbd9847af47d34434/step_07_2024_4_24_17_3_f56b9f3bf9ef483cbd9847af47d34434-9/11_icon_IO_O0_AM.png
try:
    _c11 = get_crop(11, 144, 144)
    canvas.paste(_c11, (1116, 108), _c11)
except Exception:
    pass
layout["IO:O0_AM"] = [1116, 108, 1260, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f56b9f3bf9ef483cbd9847af47d34434/step_07_2024_4_24_17_3_f56b9f3bf9ef483cbd9847af47d34434-9/12_icon_5.10_my.png
try:
    _c12 = get_crop(12, 61, 66)
    canvas.paste(_c12, (181, 1), _c12)
except Exception:
    pass
layout["5.10_my"] = [181, 1, 242, 67]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f56b9f3bf9ef483cbd9847af47d34434/step_07_2024_4_24_17_3_f56b9f3bf9ef483cbd9847af47d34434-9/13_icon_IO_O0_AM.png
try:
    _c13 = get_crop(13, 293, 144)
    canvas.paste(_c13, (144, 1290), _c13)
except Exception:
    pass
layout["IO:O0_AM"] = [144, 1290, 437, 1434]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f56b9f3bf9ef483cbd9847af47d34434/step_07_2024_4_24_17_3_f56b9f3bf9ef483cbd9847af47d34434-9/14_icon_Free.png
try:
    _c14 = get_crop(14, 134, 107)
    canvas.paste(_c14, (101, 2574), _c14)
except Exception:
    pass
layout["Free"] = [101, 2574, 235, 2681]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f56b9f3bf9ef483cbd9847af47d34434/step_07_2024_4_24_17_3_f56b9f3bf9ef483cbd9847af47d34434-9/15_icon_icon_15.png
try:
    _c15 = get_crop(15, 98, 60)
    canvas.paste(_c15, (1217, 3), _c15)
except Exception:
    pass
layout["icon_15"] = [1217, 3, 1315, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f56b9f3bf9ef483cbd9847af47d34434/step_07_2024_4_24_17_3_f56b9f3bf9ef483cbd9847af47d34434-9/16_icon_icon_16.png
try:
    _c16 = get_crop(16, 54, 65)
    canvas.paste(_c16, (247, 1), _c16)
except Exception:
    pass
layout["icon_16"] = [247, 1, 301, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f56b9f3bf9ef483cbd9847af47d34434/step_07_2024_4_24_17_3_f56b9f3bf9ef483cbd9847af47d34434-9/17_icon_icon_17.png
try:
    _c17 = get_crop(17, 65, 65)
    canvas.paste(_c17, (308, 1), _c17)
except Exception:
    pass
layout["icon_17"] = [308, 1, 373, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f56b9f3bf9ef483cbd9847af47d34434/step_07_2024_4_24_17_3_f56b9f3bf9ef483cbd9847af47d34434-9/18_icon_icon_18.png
try:
    _c18 = get_crop(18, 57, 67)
    canvas.paste(_c18, (1317, 0), _c18)
except Exception:
    pass
layout["icon_18"] = [1317, 0, 1374, 67]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f56b9f3bf9ef483cbd9847af47d34434/step_07_2024_4_24_17_3_f56b9f3bf9ef483cbd9847af47d34434-9/19_text_5.10_my.png
try:
    _c19 = get_crop(19, 149, 45)
    canvas.paste(_c19, (22, 15), _c19)
except Exception:
    pass
layout["5.10_my"] = [22, 15, 171, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f56b9f3bf9ef483cbd9847af47d34434/step_07_2024_4_24_17_3_f56b9f3bf9ef483cbd9847af47d34434-9/20_text_About_this_event.png
try:
    _c20 = get_crop(20, 454, 61)
    canvas.paste(_c20, (45, 2080), _c20)
except Exception:
    pass
layout["About_this_event"] = [45, 2080, 499, 2141]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f56b9f3bf9ef483cbd9847af47d34434/step_07_2024_4_24_17_3_f56b9f3bf9ef483cbd9847af47d34434-9/21_text_Hobbies_Special_Interest.png
try:
    _c21 = get_crop(21, 496, 55)
    canvas.paste(_c21, (86, 2192), _c21)
except Exception:
    pass
layout["Hobbies_&_Special_Interes"] = [86, 2192, 582, 2247]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f56b9f3bf9ef483cbd9847af47d34434/step_07_2024_4_24_17_3_f56b9f3bf9ef483cbd9847af47d34434-9/22_text_Anime_Comics.png
try:
    _c22 = get_crop(22, 288, 50)
    canvas.paste(_c22, (597, 2192), _c22)
except Exception:
    pass
layout["Anime_Comics"] = [597, 2192, 885, 2242]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f56b9f3bf9ef483cbd9847af47d34434/step_07_2024_4_24_17_3_f56b9f3bf9ef483cbd9847af47d34434-9/23_text_General_Admission.png
try:
    _c23 = get_crop(23, 415, 55)
    canvas.paste(_c23, (116, 2451), _c23)
except Exception:
    pass
layout["General_Admission"] = [116, 2451, 531, 2506]
