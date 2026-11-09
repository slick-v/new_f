import uuid


class IngestError(Exception):
    """The uploaded file can't be turned into chunks (bad type, unreadable, no text)."""


class DuplicateDocumentError(Exception):
    def __init__(self, document_id: uuid.UUID):
        self.document_id = document_id
        super().__init__(f"Document already uploaded: {document_id}")