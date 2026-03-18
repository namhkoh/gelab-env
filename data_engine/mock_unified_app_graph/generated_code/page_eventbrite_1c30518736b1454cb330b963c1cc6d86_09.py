# page_id: page_eventbrite_1c30518736b1454cb330b963c1cc6d86_09
# screenshot: 2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-11.png
# step_index: 9/9
# task: Open Eventbrite. Search for "Open Mic Nights". Filter the results to only include free events. Select the first non-promoted event in the list - what"s the location of that event?
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw background and structural UI elements for the mobile event page layout.
# Uses provided variables: canvas (PIL Image), draw (ImageDraw), font_sm, font_md, font_lg, font_xl

W, H = canvas.size

# Colors
bg_color = (250, 250, 252)          # subtle off-white page background
status_bar_color = (200, 200, 200)  # top status bar light gray
hero_top = (245, 190, 80)           # hero gradient top (warm gold)
hero_bottom = (135, 60, 20)         # hero gradient bottom (deeper amber)
card_bg = (247, 249, 251)           # light card background
card_shadow = (220, 220, 225)       # shadow tone
divider_color = (235, 235, 238)     # thin divider lines
muted_bg = (250, 251, 253)          # very subtle section background
accent_blue = (54, 85, 247)         # selection accent

# Fill overall page background
draw.rectangle((0, 0, W, H), fill=bg_color)

# Status bar area (top)
status_h = 72
draw.rectangle((0, 0, W, status_h), fill=status_bar_color)

# Hero/banner area - vertical gradient rectangle beneath status bar
hero_top_y = status_h
hero_bottom_y = status_h + 360
for y in range(hero_top_y, hero_bottom_y):
    t = (y - hero_top_y) / max(1, (hero_bottom_y - hero_top_y - 1))
    r = int(hero_top[0] * (1 - t) + hero_bottom[0] * t)
    g = int(hero_top[1] * (1 - t) + hero_bottom[1] * t)
    b = int(hero_top[2] * (1 - t) + hero_bottom[2] * t)
    draw.line((0, y, W, y), fill=(r, g, b))

# Soft dark overlay along left/right edges for framing (subtle)
edge_shade = (0, 0, 0)
for i in range(30):
    alpha = int(6 - i * 0.18)  # fake alpha by blending to background color
    if alpha <= 0:
        break
    shade = (
        int(hero_bottom[0] * (1 - alpha / 12) + edge_shade[0] * (alpha / 12)),
        int(hero_bottom[1] * (1 - alpha / 12) + edge_shade[1] * (alpha / 12)),
        int(hero_bottom[2] * (1 - alpha / 12) + edge_shade[2] * (alpha / 12)),
    )
    # left edge
    draw.rectangle((0 + i, hero_top_y, 40 + i, hero_bottom_y), fill=shade)
    # right edge
    draw.rectangle((W - 40 - i, hero_top_y, W - i, hero_bottom_y), fill=shade)

# Divider under hero (thin)
draw.rectangle((40, hero_bottom_y + 18, W - 40, hero_bottom_y + 20), fill=divider_color)

# Organizer "card" background (rounded) under title area
org_card_top = hero_bottom_y + 40
org_card_left = 48
org_card_right = W - 48
org_card_bottom = org_card_top + 120
# shadow
draw.rounded_rectangle(
    (org_card_left + 4, org_card_top + 8, org_card_right + 4, org_card_bottom + 8),
    radius=28, fill=card_shadow
)
# main card
draw.rounded_rectangle(
    (org_card_left, org_card_top, org_card_right, org_card_bottom),
    radius=28, fill=card_bg
)

# Small subtle horizontal divider below organizer card area
draw.rectangle((40, org_card_bottom + 36, W - 40, org_card_bottom + 38), fill=divider_color)

# Info list area (location, duration, refund) - just the structural vertical spacing + divider
info_top = org_card_bottom + 62
# draw a faint background band for the info area
draw.rectangle((0, info_top - 8, W, info_top + 240), fill=muted_bg)
# thin divider below info
draw.rectangle((40, info_top + 240, W - 40, info_top + 242), fill=divider_color)

# "Select date and time" cards container area
dates_top = info_top + 300
dates_height = 320
dates_left = 36
dates_right = W - 36
# section title area spacing (no text)
draw.rectangle((0, dates_top - 64, W, dates_top - 62), fill=(0, 0, 0, 0))

# Draw three date cards (structural backgrounds and outlines)
card_w = 360
card_h = 260
gap = 36
start_x = dates_left
card_y1 = dates_top
card_y2 = card_y1 + card_h

# Subtle outer container divider above date cards
draw.rectangle((40, dates_top - 30, W - 40, dates_top - 28), fill=divider_color)

for i in range(3):
    x1 = start_x + i * (card_w + gap)
    x2 = x1 + card_w
    # shadow
    draw.rounded_rectangle((x1 + 6, card_y1 + 8, x2 + 6, card_y2 + 8), radius=20, fill=card_shadow)
    # card background
    draw.rounded_rectangle((x1, card_y1, x2, card_y2), radius=20, fill=(255, 255, 255))
    # subtle border
    draw.rounded_rectangle((x1, card_y1, x2, card_y2), radius=20, outline=(240, 240, 244), width=2)

# Highlight the first card with accent border (selected state)
sel_x1 = start_x
sel_x2 = sel_x1 + card_w
draw.rounded_rectangle((sel_x1 - 6, card_y1 - 6, sel_x2 + 6, card_y2 + 6), radius=26, outline=accent_blue, width=6)

# Divider line below date cards
draw.rectangle((40, card_y2 + 28, W - 40, card_y2 + 30), fill=divider_color)

# "About this event" section background band (structural only)
about_top = card_y2 + 72
about_bottom = about_top + 320
draw.rectangle((0, about_top, W, about_bottom), fill=bg_color)
# subtle rounded white card behind the content region (container)
about_left = 40
about_right = W - 40
draw.rounded_rectangle((about_left, about_top + 28, about_right, about_top + 160), radius=18, fill=(255, 255, 255), outline=(245, 245, 248))

# Large bottom divider prior to description area
draw.rectangle((40, about_top + 200, W - 40, about_top + 202), fill=divider_color)

# Final subtle footer background band
footer_top = H - 220
draw.rectangle((0, footer_top, W, H), fill=muted_bg)

# Note: do not draw any icons or text — only layout/background structure has been created.

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_09_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-11/00_icon_May.png
try:
    _c0 = get_crop(0, 450, 516)
    canvas.paste(_c0, (24, 1972), _c0)
except Exception:
    pass
layout["May"] = [24, 1972, 474, 2488]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_09_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-11/01_icon_Follow.png
try:
    _c1 = get_crop(1, 331, 144)
    canvas.paste(_c1, (1013, 1068), _c1)
except Exception:
    pass
layout["Follow"] = [1013, 1068, 1344, 1212]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_09_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-11/02_icon_June.png
try:
    _c2 = get_crop(2, 450, 516)
    canvas.paste(_c2, (474, 1972), _c2)
except Exception:
    pass
layout["June"] = [474, 1972, 924, 2488]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_09_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-11/03_icon_FiRe.png
try:
    _c3 = get_crop(3, 144, 144)
    canvas.paste(_c3, (1116, 108), _c3)
except Exception:
    pass
layout["FiRe"] = [1116, 108, 1260, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_09_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-11/04_icon_Performing_Visual_Arts.png
try:
    _c4 = get_crop(4, 541, 97)
    canvas.paste(_c4, (39, 2735), _c4)
except Exception:
    pass
layout["Performing_&_Visual_Arts"] = [39, 2735, 580, 2832]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_09_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-11/05_icon_Share.png
try:
    _c5 = get_crop(5, 144, 144)
    canvas.paste(_c5, (1260, 108), _c5)
except Exception:
    pass
layout["Share"] = [1260, 108, 1404, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_09_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-11/06_icon_4.54_Wy.png
try:
    _c6 = get_crop(6, 144, 144)
    canvas.paste(_c6, (36, 108), _c6)
except Exception:
    pass
layout["4.54_Wy"] = [36, 108, 180, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_09_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-11/07_icon_South.png
try:
    _c7 = get_crop(7, 699, 144)
    canvas.paste(_c7, (144, 1028), _c7)
except Exception:
    pass
layout["South"] = [144, 1028, 843, 1172]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_09_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-11/08_icon_5.00_PM.png
try:
    _c8 = get_crop(8, 450, 516)
    canvas.paste(_c8, (924, 1972), _c8)
except Exception:
    pass
layout["5.00_PM"] = [924, 1972, 1374, 2488]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_09_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-11/09_icon_July.png
try:
    _c9 = get_crop(9, 450, 516)
    canvas.paste(_c9, (924, 1972), _c9)
except Exception:
    pass
layout["July"] = [924, 1972, 1374, 2488]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_09_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-11/10_icon_4.54_Wy.png
try:
    _c10 = get_crop(10, 63, 69)
    canvas.paste(_c10, (180, 0), _c10)
except Exception:
    pass
layout["4.54_Wy"] = [180, 0, 243, 69]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_09_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-11/11_icon_NIGHHT.png
try:
    _c11 = get_crop(11, 144, 144)
    canvas.paste(_c11, (1116, 108), _c11)
except Exception:
    pass
layout["NIGHHT"] = [1116, 108, 1260, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_09_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-11/12_icon_4.54_Wy.png
try:
    _c12 = get_crop(12, 62, 69)
    canvas.paste(_c12, (114, 0), _c12)
except Exception:
    pass
layout["4.54_Wy"] = [114, 0, 176, 69]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_09_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-11/13_icon_icon_13.png
try:
    _c13 = get_crop(13, 56, 65)
    canvas.paste(_c13, (1317, 0), _c13)
except Exception:
    pass
layout["icon_13"] = [1317, 0, 1373, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_09_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-11/14_icon_icon_14.png
try:
    _c14 = get_crop(14, 100, 63)
    canvas.paste(_c14, (1214, 1), _c14)
except Exception:
    pass
layout["icon_14"] = [1214, 1, 1314, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_09_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-11/15_icon_icon_15.png
try:
    _c15 = get_crop(15, 55, 67)
    canvas.paste(_c15, (246, 1), _c15)
except Exception:
    pass
layout["icon_15"] = [246, 1, 301, 68]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_09_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-11/16_icon_icon_16.png
try:
    _c16 = get_crop(16, 64, 65)
    canvas.paste(_c16, (309, 1), _c16)
except Exception:
    pass
layout["icon_16"] = [309, 1, 373, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_09_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-11/17_text_LoS_ANGELES_LGBT_OCNTER.png
try:
    _c17 = get_crop(17, 258, 27)
    canvas.paste(_c17, (636, 113), _c17)
except Exception:
    pass
layout["LoS_ANGELES_LGBT_OCNTER_%"] = [636, 113, 894, 140]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_09_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-11/18_text_Los_Angeles_LGBT_Center.png
try:
    _c18 = get_crop(18, 1344, 144)
    canvas.paste(_c18, (48, 1295), _c18)
except Exception:
    pass
layout["Los_Angeles_LGBT_Center"] = [48, 1295, 1392, 1439]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_09_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-11/19_text_Center_South.png
try:
    _c19 = get_crop(19, 1344, 144)
    canvas.paste(_c19, (48, 1295), _c19)
except Exception:
    pass
layout["Center_South"] = [48, 1295, 1392, 1439]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_09_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-11/20_text_2_hrs.png
try:
    _c20 = get_crop(20, 112, 50)
    canvas.paste(_c20, (141, 1452), _c20)
except Exception:
    pass
layout["2_hrs"] = [141, 1452, 253, 1502]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_09_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-11/21_text_Refund_policy.png
try:
    _c21 = get_crop(21, 299, 63)
    canvas.paste(_c21, (138, 1558), _c21)
except Exception:
    pass
layout["Refund_policy"] = [138, 1558, 437, 1621]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_09_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-11/22_text_The_organizer_will_review_refund_request.png
try:
    _c22 = get_crop(22, 1344, 144)
    canvas.paste(_c22, (48, 1295), _c22)
except Exception:
    pass
layout["The_organizer_will_review"] = [48, 1295, 1392, 1439]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_09_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-11/23_text_Select_date_and_time.png
try:
    _c23 = get_crop(23, 450, 516)
    canvas.paste(_c23, (24, 1972), _c23)
except Exception:
    pass
layout["Select_date_and_time"] = [24, 1972, 474, 2488]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_09_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-11/24_text_About_this_event.png
try:
    _c24 = get_crop(24, 452, 65)
    canvas.paste(_c24, (45, 2645), _c24)
except Exception:
    pass
layout["About_this_event"] = [45, 2645, 497, 2710]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_09_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-11/25_text_Come_share_your_talents_and_watch_others.png
try:
    _c25 = get_crop(25, 450, 516)
    canvas.paste(_c25, (474, 1972), _c25)
except Exception:
    pass
layout["Come_share_your_talents_a"] = [474, 1972, 924, 2488]
