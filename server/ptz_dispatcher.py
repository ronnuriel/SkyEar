from shared.event_schema import FusedAlert

def dispatch_ptz_for_alert(alert: FusedAlert):
    # Camera-only visual confirmation placeholder.
    if alert.level >= 2:
        print("[PTZ] candidate alert; dispatch camera for visual confirmation only")
