"""Compatibility shim for protobuf MessageFactory.GetPrototype

Some installed packages expect MessageFactory.GetPrototype which is missing
in newer protobuf releases. This module adds a fallback implementation that
tries to resolve a message class by descriptor full name.

Import this early in your application (before imports that may use MessageFactory).
"""
from google.protobuf import message_factory as _mf
from google.protobuf import symbol_database as _symdb

if _mf is not None:
    if not hasattr(_mf.MessageFactory, 'GetPrototype'):
        def _compat_GetPrototype(self, descriptor):
            """Compatibility GetPrototype implementation.

            Tries several strategies to obtain a Message class from a Descriptor:
            1. MessageFactory.GetMessageClass(full_name) (newer API)
            2. Default symbol database GetPrototype(descriptor) (if available)
            3. Raise AttributeError to signal incompatibility
            """
            # Try newer API first
            get_msg_cls = getattr(self, 'GetMessageClass', None)
            if callable(get_msg_cls):
                cls = get_msg_cls(descriptor.full_name)
                if cls is not None:
                    return cls

            # Try symbol database
            db = _symdb.Default()
            get_proto = getattr(db, 'GetPrototype', None)
            if callable(get_proto):
                return get_proto(descriptor)

            # As a last resort, try the module-level GetPrototype if present
            module_get = getattr(_mf, 'GetPrototype', None)
            if callable(module_get):
                return module_get(descriptor)

            raise AttributeError('GetPrototype not available on MessageFactory and no compatible fallback found')

        # Attach compatibility method
        setattr(_mf.MessageFactory, 'GetPrototype', _compat_GetPrototype)
        # Also ensure SymbolDatabase has GetPrototype (some libs call Default().GetPrototype)
        if _symdb is not None and not hasattr(_symdb.SymbolDatabase, 'GetPrototype'):
            def _symdb_GetPrototype(self, descriptor):
                # Prefer module-level GetPrototype if available
                module_get = getattr(_mf, 'GetPrototype', None)
                if callable(module_get):
                    return module_get(descriptor)

                # Fallback to MessageFactory instance
                mf_inst = _mf.MessageFactory()
                if hasattr(mf_inst, 'GetPrototype'):
                    return mf_inst.GetPrototype(descriptor)
                # Newer API: GetMessageClass by full name
                get_msg_cls = getattr(mf_inst, 'GetMessageClass', None)
                if callable(get_msg_cls):
                    cls = get_msg_cls(descriptor.full_name)
                    if cls is not None:
                        return cls

                # Last resort: try symbol database Default() lookups
                db = _symdb.Default()
                if hasattr(db, '_GetPrototype'):
                    return db._GetPrototype(descriptor)

                raise AttributeError('SymbolDatabase.GetPrototype not available and no fallback found')

            setattr(_symdb.SymbolDatabase, 'GetPrototype', _symdb_GetPrototype)
