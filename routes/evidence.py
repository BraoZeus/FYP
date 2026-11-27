"""
Evidence Blueprint — secure evidence serving with decryption,
integrity verification, and audit logging.

Evidence images are AES-256-GCM encrypted on disk. This blueprint
decrypts them in-memory and serves to authenticated users only,
preventing direct static file access to sensitive captures.
"""
import os
from io import BytesIO

from flask import Blueprint, send_file, abort, current_app
from flask_login import login_required, current_user

from database import db, ViolationLog, EmergencyAlert
from crypto_utils import decrypt_evidence, verify_evidence_integrity
from audit_utils import log_audit_event, ACTION_EVIDENCE_VIEW, ACTION_EVIDENCE_INTEGRITY_FAIL
from auth_utils import role_required, ROLE_ADMIN, ROLE_AFM

evidence_bp = Blueprint('evidence', __name__)


@evidence_bp.route('/evidence/<int:evidence_id>')
@role_required(ROLE_ADMIN, ROLE_AFM)
def serve_evidence_by_id(evidence_id):
    """Lookup evidence image by ViolationLog or EmergencyAlert ID (Admin & AFM only)."""
    v = db.session.get(ViolationLog, evidence_id)
    if v and v.image_path:
        return serve_evidence(v.image_path)
    a = db.session.get(EmergencyAlert, evidence_id)
    if a and a.image_path:
        return serve_evidence(a.image_path)
    abort(404)


@evidence_bp.route('/evidence/<path:filename>')
@role_required(ROLE_ADMIN, ROLE_AFM)
def serve_evidence(filename):
    """
    Serve an encrypted evidence image after decryption and integrity check.

    This route replaces direct static file access for evidence images.
    It decrypts the AES-256-GCM encrypted file in memory and streams
    the plaintext JPEG to authenticated users only.

    If the file has a stored SHA-256 hash, the decrypted data is verified
    against it to detect tampering.
    """
    static_dir = current_app.static_folder or os.path.join(
        current_app.root_path, 'static'
    )

    # The DB stores paths like "evidence/violation_xxx.jpg.enc".
    # The route URL is /evidence/<path:filename>, so filename arrives
    # as the full DB value including the "evidence/" prefix.
    # Strip the prefix if present, then resolve to just the base filename
    # for safe path construction (prevents path traversal).
    clean_name = filename
    if clean_name.startswith('evidence/'):
        clean_name = clean_name[len('evidence/'):]
    clean_name = os.path.basename(clean_name)  # prevent path traversal

    abs_path = os.path.join(static_dir, 'evidence', clean_name)

    if not os.path.isfile(abs_path):
        # Try with .enc extension appended for legacy references
        abs_path_enc = abs_path + '.enc'
        if os.path.isfile(abs_path_enc):
            abs_path = abs_path_enc
            clean_name = clean_name + '.enc'
        else:
            abort(404)

    # Look up evidence record for integrity hash
    rel_path = f"evidence/{os.path.basename(clean_name)}"
    evidence_hash = None
    resource_type = None
    resource_id = None

    # Search in ViolationLog first, then EmergencyAlert
    record = ViolationLog.query.filter_by(image_path=rel_path).first()
    if record:
        evidence_hash = record.evidence_hash
        resource_type = 'violation'
        resource_id = record.id
    else:
        record = EmergencyAlert.query.filter_by(image_path=rel_path).first()
        if record:
            evidence_hash = record.evidence_hash
            resource_type = 'alert'
            resource_id = record.id

    if abs_path.endswith('.enc'):
        # Decrypt encrypted evidence
        try:
            with open(abs_path, 'rb') as f:
                encrypted_data = f.read()
            plaintext_data = decrypt_evidence(encrypted_data)
        except Exception:
            abort(500)

        # Verify integrity hash if available
        if evidence_hash and not verify_evidence_integrity(
            plaintext_data, evidence_hash
        ):
            log_audit_event(
                ACTION_EVIDENCE_INTEGRITY_FAIL,
                resource_type=resource_type,
                resource_id=resource_id,
                details=f'Integrity check FAILED for {rel_path}',
                success=False,
            )
            abort(403)

        # Log evidence access
        log_audit_event(
            ACTION_EVIDENCE_VIEW,
            resource_type=resource_type,
            resource_id=resource_id,
            details=f'Accessed encrypted evidence: {rel_path}',
        )

        return send_file(
            BytesIO(plaintext_data),
            mimetype='image/jpeg',
            download_name=os.path.basename(clean_name).replace('.enc', ''),
        )

    else:
        # Legacy unencrypted file — serve directly but still audit
        log_audit_event(
            ACTION_EVIDENCE_VIEW,
            resource_type=resource_type,
            resource_id=resource_id,
            details=f'Accessed legacy unencrypted evidence: {rel_path}',
        )
        return send_file(abs_path, mimetype='image/jpeg')
